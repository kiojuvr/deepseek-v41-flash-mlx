#include "dsv41/expert_backing.hpp"
#include <iostream>
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: dsv41-repair-expert-backing root");
 dsv41::repair_expert_backing_identity(argv[1]);
 std::cout<<"PASS: repaired only post-rename expert-backing ctime identities\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
