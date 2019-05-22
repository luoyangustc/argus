#include "censor.hpp"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <fstream>
#include <streambuf>
#include <iostream>

using namespace std;
using namespace qc;
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "./demo <image_path>"<< endl;
        return -1;
    }

    pic_info_t meta;
    meta.pid = 1001;
    strcpy(meta.zip_name,"test001_meta");
    strcpy(meta.bcp_name,"test001_bcp");
    strcpy(meta.data_id,"tesqt001_data");
    strcpy(meta.pic_name,"test001_pic");
    meta.line = 0;

    ifstream in(argv[1]);
    std::string file((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    in.close();
    
    Censor censor;

    censor.init(10);

    while(true){
        int ret = censor.eval((void*)file.c_str(),file.size(), &meta);
        if (ret == ErrTooManyRequests)
        {
            usleep(1000000);
            continue;
        }
        if ( ret != 0){
            cout << "ret:" << ret << endl;
            return ret;
        }
        meta.line++;
        if (meta.line % 100 == 0) {
            metrics_info_t info = censor.metrics();
            cout << "metrics total:" << info.total << ", done: " << info.done 
                 << ", waiting: " << info.waiting << ", censor: "<< info.censor
                 << ", skip: " << info.skip << ", error: " << info.error
                 << ", last_qps:" << info.last_qps << ", last_filter_rate: "<< info.last_filter_rate
                 << ", qps: " << info.qps << ", filter_rate: " << info.filter_rate
                 << ", normal: " << info.normal << ", pulp: " << info.pulp 
                 << ", terror:" << info.terror << ", politicion:"<< info.politicion 
                 << ", march: "<< info.march << ", text: "<< info.text 
                 << ", runtime: "<< info.runtime << endl;
        }
    }
    return 0;
}