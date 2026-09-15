#include "dsv41/swa_state.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace dsv41;
void check(bool b){if(!b)throw std::runtime_error("SWA contract failed");}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed);}
int main(){try{
 for(std::size_t length:{1,127,128,129,255,256,257,1025}){
  std::vector<std::uint16_t> rows(length*512);
  for(std::size_t t=0;t<length;++t)std::fill_n(rows.begin()+t*512,512,std::uint16_t(0x3f00+t%127));
  SwaReferenceState full,incremental;full.append(rows,0);
  for(std::size_t t=0;t<length;++t)incremental.append(std::span(rows).subspan(t*512,512),t);
  check(full.chronological_rows()==incremental.chronological_rows());
  auto expected=std::vector<std::uint16_t>(rows.end()-std::min(length,std::size_t(128))*512,rows.end());
  check(full.chronological_rows()==expected);
  auto indices=full.official_decode_indices();std::vector<std::uint16_t> selected;
  for(auto i:indices)if(i>=0)selected.insert(selected.end(),full.physical_rows().begin()+i*512,full.physical_rows().begin()+(i+1)*512);
  check(selected==expected);
  auto fork=full;auto saved=full.chronological_rows();std::vector<std::uint16_t> row(512,0x4000);
  fork.append(row,length);check(full.chronological_rows()==saved&&full.position()==length);
  rejects([&]{full.append(row,0);});check(full.position()==length);
  row.back()=0x7fc0;rejects([&]{full.append(row,length);});check(full.chronological_rows()==saved);
  full.reset();check(full.position()==0&&full.chronological_rows().empty());rejects([&]{full.official_decode_indices();});
 }
 SwaReferenceState empty;rejects([&]{empty.append({},0);});std::vector<std::uint16_t> malformed(513);rejects([&]{empty.append(malformed,0);});
 // Self-backed input must behave like an owned snapshot even at a wrapped cursor.
 std::vector<std::uint16_t> seed(129*512);
 for(std::size_t t=0;t<129;++t)std::fill_n(seed.begin()+t*512,512,std::uint16_t(0x3f00+t%127));
 SwaReferenceState aliased;aliased.append(seed,0);auto independent=aliased;
 std::vector<std::uint16_t> snapshot(aliased.physical_rows().begin(),aliased.physical_rows().end());
 aliased.append(aliased.physical_rows(),129);independent.append(snapshot,129);
 check(aliased.chronological_rows()==independent.chronological_rows());
 std::cout<<"SWA wrap, chunk/token equivalence, official index order, fork/reset and atomic rejection passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
