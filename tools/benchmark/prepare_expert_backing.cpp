#include "dsv41/expert_backing.hpp"
#include <iostream>
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("usage: dsv41-prepare-expert-backing checkpoint m1-summary destination");
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 dsv41::prepare_expert_backing(catalog,argv[3]);
 std::cout<<"PASS: complete file-backed expert atlas published; runtime qualification still required\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
