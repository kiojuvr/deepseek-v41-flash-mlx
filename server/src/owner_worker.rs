//! Keep thread-affine native state on one OS thread, including construction/drop.
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

struct Job<Q, R> {
    request: Q,
    reply: SyncSender<Result<R, String>>,
}

pub struct OwnerWorker<Q, R> {
    sender: Option<SyncSender<Job<Q, R>>>,
    thread: Option<JoinHandle<()>>,
}

impl<Q: Send + 'static, R: Send + 'static> OwnerWorker<Q, R> {
    // T intentionally has no Send bound: it is never transferred between threads.
    pub fn start<T, F, H>(create: F, mut handle: H) -> Result<Self, String>
    where
        F: FnOnce() -> Result<T, String> + Send + 'static,
        H: FnMut(&mut T, Q) -> R + Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel::<Job<Q, R>>(1);
        let (ready, startup) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("dsv41-model".into())
            .spawn(move || {
                let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(create));
                let mut state = match created {
                    Ok(Ok(state)) => state,
                    Ok(Err(error)) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                    Err(_) => {
                        let _ = ready.send(Err("native model initialization panicked".into()));
                        return;
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                while let Ok(job) = receiver.recv() {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        handle(&mut state, job.request)
                    })) {
                        Ok(result) => {
                            let _ = job.reply.send(Ok(result));
                        }
                        Err(_) => {
                            let _ = job.reply.send(Err("native owner thread panicked".into()));
                            // Do not reuse potentially inconsistent native state after a panic.
                            break;
                        }
                    }
                }
                // state drops here, on the same thread that constructed it.
            })
            .map_err(|e| e.to_string())?;
        match startup.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                thread: Some(thread),
            }),
            result => {
                drop(sender);
                let _ = thread.join();
                Err(match result {
                    Ok(Err(error)) => error,
                    _ => "native owner exited during startup".into(),
                })
            }
        }
    }

    pub fn call(&self, request: Q) -> Result<R, String> {
        let (reply, result) = mpsc::sync_channel(1);
        self.sender
            .as_ref()
            .ok_or("native owner is shut down")?
            .send(Job { request, reply })
            .map_err(|_| "native owner is unavailable")?;
        result
            .recv()
            .map_err(|_| "native owner exited without a result")?
    }
}

impl<Q, R> Drop for OwnerWorker<Q, R> {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    struct Affine {
        owner: thread::ThreadId,
        _not_send: Rc<()>,
        dropped: SyncSender<thread::ThreadId>,
    }
    impl Drop for Affine {
        fn drop(&mut self) {
            self.dropped.send(thread::current().id()).unwrap();
        }
    }
    #[test]
    fn construct_calls_from_different_threads_and_drop_share_owner() {
        let (dropped, drop_result) = mpsc::sync_channel(1);
        let worker = OwnerWorker::start(
            move || {
                Ok(Affine {
                    owner: thread::current().id(),
                    _not_send: Rc::new(()),
                    dropped,
                })
            },
            |state, ()| {
                assert_eq!(state.owner, thread::current().id());
                state.owner
            },
        )
        .unwrap();
        let owner = worker.call(()).unwrap();
        assert_ne!(owner, thread::current().id());
        thread::scope(|scope| {
            scope
                .spawn(|| assert_eq!(worker.call(()).unwrap(), owner))
                .join()
                .unwrap();
        });
        drop(worker);
        assert_eq!(drop_result.recv().unwrap(), owner);
    }
    #[test]
    fn startup_error_and_panic_do_not_hang() {
        assert!(
            OwnerWorker::<(), ()>::start(|| Err::<(), _>("load failed".into()), |_, _| ()).is_err()
        );
        assert!(
            OwnerWorker::<(), ()>::start(
                || -> Result<(), String> { panic!("load panic") },
                |_, _| ()
            )
            .is_err()
        );
    }
    #[test]
    fn request_panic_reports_failure_and_drops_state() {
        let worker =
            OwnerWorker::start(|| Ok(()), |_, ()| -> () { panic!("request panic") }).unwrap();
        assert!(worker.call(()).is_err());
        assert!(worker.call(()).is_err());
    }
}
