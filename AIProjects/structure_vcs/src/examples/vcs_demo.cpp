#include <cstdio>
#include <future>
#include <fstream>
#include <sstream>
#include "common/log.hpp"
#include "document.h"       // rapidjso
#include "vcs.hpp"

void mock_callback(const char* json_str, const char* channel_id) {
    LOG(INFO) << "channel_id: " << channel_id << " json_str_len: " << strlen(json_str);
    return;
}

int main(int argc, char* argv[]) {
    int status_code = 0;

    if ( argc < 3 ) {
        LOG(FATAL) << "Usage: vss_demo <init_json_filepath> <start_analysis_filepath>";
        return -1;
    }
    auto init_conf_file = argv[1];
    auto start_analysis_file = argv[2];
    rapidjson::Document init_param_doc, add_param_doc;

    std::ifstream init_file;
    init_file.open(init_conf_file);
    std::stringstream init_file_stream;
    init_file_stream << init_file.rdbuf(); //read the file
    if ( init_param_doc.Parse(init_file_stream.str().c_str()).HasParseError() ) {
        LOG(FATAL) << "parse init_config file error";
        return -1;
    }

    std::ifstream add_file;
    add_file.open(start_analysis_file);
    std::stringstream add_file_stream;
    add_file_stream << add_file.rdbuf();
    if ( add_param_doc.Parse(add_file_stream.str().c_str()).HasParseError() ) {
        LOG(FATAL) << "parse add_config file error";
        return -1;
    }

    LOG(INFO) << "initAnalysisModule";
    status_code = initAnalysisModule(init_file_stream.str().c_str(), init_file_stream.str().length());
    if ( status_code != status_Success ) {
        LOG(FATAL) << "initAnalysisModule failed, status_code: " << status_code;
        return -1;
    }
    LOG(INFO) << "initAnalysisModule succ";

    LOG(INFO) << "startVidAnalysis";
    if ( add_param_doc.IsArray() ) {
        auto arr = add_param_doc.GetArray();
        for (const auto& ele: add_param_doc.GetArray()) {
            status_code = startVidAnalysis(
                ele["url"].GetString(),
                ele["url"].GetStringLength(),
                ele["channelId"].GetString(),
                ele["channelId"].GetStringLength(),
                mock_callback);
            if (status_code != status_Success)
            {
                LOG(FATAL) << "startVidAnalysis failed, status_code: " << status_code;
                return -1;
            }
        }
    } else {
        status_code = startVidAnalysis(
            add_param_doc["url"].GetString(),
            add_param_doc["url"].GetStringLength(),
            add_param_doc["channelId"].GetString(),
            add_param_doc["channelId"].GetStringLength(),
            mock_callback
        );
        if ( status_code != status_Success ) {
            LOG(FATAL) << "startVidAnalysis failed, status_code: " << status_code;
            return -1;
        }
    }
    LOG(INFO) << "startVidAnalysis succ, wait forever";

    std::promise<void>().get_future().wait();
    
    return 0;
}

