#include <cstddef>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include "proto/inference.pb.h"
#include "common/log.hpp"
#include "document.h"       // rapidjson
#include "stringbuffer.h"
#include "writer.h"
#include "vss.hpp"
#include "vcs.hpp"
#include "callback.hpp"
#include "util.hpp"

void *g_vss_instance = nullptr;
callback::CallbackWorker *g_callback_worker = nullptr;

const char* g_alg_conf = R"(
{
    "debug_msg": false,

    "person_vehicle_det_local_model": "/data/models/yolov3-tiny_pvn.tronmodel",
    "batch_size": 1,

    "class_index_person": 1,
    "class_index_motor": 2,
    "class_index_non_motor": 3,

    "person_attribute_local_model": "/data/models/resnet18_personattr_merged.tronmodel",

    "vehicle_attribute_local_model": "/data/models/vehicle_attribute_168_iter_50000_merged.tronmodel",

    "plate_det_local_model":"/data/models/east_plate_320X320_P4_v2.tronmodel",
    "plate_recog_local_model":"/data/models/crnn_13_4_merged.tronmodel"
}
)";

statusCode initAnalysisModule(const char* json_str, int json_str_len) {
    LOG(INFO) << "initAnalysisModule, params: " << std::endl << json_str;
    // vss
    int code = 0;
    char* err = nullptr;
    inference::CreateParams create_params;

    rapidjson::Document vcs_param_doc, alg_params_doc;
    if ( vcs_param_doc.Parse(json_str).HasParseError() ) {
        LOG(WARNING) << "parse initParams error";
        return status_ErrInputParam;
    }
    if ( alg_params_doc.Parse(g_alg_conf).HasParseError() ) {
        LOG(WARNING) << "parse g_alg_conf error";
        return status_ErrInputParam;
    }
    // merge vcs_params_doc and alg_parms_doc into create_params
    util::merge_docs(vcs_param_doc, alg_params_doc, vcs_param_doc.GetAllocator());

    rapidjson::StringBuffer buffer;
    buffer.Clear();
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    vcs_param_doc.Accept(writer);
    std::string vcs_param_str = buffer.GetString();

    create_params.set_custom_params(vcs_param_str.c_str());
    auto create_params_size = create_params.ByteSize();
    std::vector<char> create_params_data(create_params_size, 0);
    create_params.SerializeToArray(create_params_data.data(), create_params_size);

    // init vss instance
    g_vss_instance = VSSCreate(create_params_data.data(), create_params_size, &code, &err);
    if (g_vss_instance == nullptr) {
        LOG(WARNING) << "vss Create error";
        return status_Error;
    }
    if ( code != vss_status_success ) {
        LOG(WARNING) << "vss Create code " << code << " != " << vss_status_success << " err: " << err;
        return status_Error;
    }

    g_callback_worker = new callback::CallbackWorker(g_vss_instance);
    // start thread for calling vss Process and callback
    if ( g_callback_worker->AsyncWaitAndCallback() ) {
        LOG(WARNING) << "callback::AsyncWaitAndCallback failed";
        return status_Error;
    }

    LOG(INFO) << "initAnalysisModule success";
    return status_Success;
}

statusCode uninitAnalysisModule() {
    LOG(INFO) << "uninitAnalysisModule";
    VSSRelease(g_vss_instance);
    if (g_callback_worker) {
        delete g_callback_worker;
    }

    return status_Success;
}

statusCode startVidAnalysis(
    const char* url, int url_len,
    const char* channel_id, int channel_id_len,
    vid_analysis_cb_func algo_callback) {
    LOG(INFO) << "startVidAnalysis, url: " << url << " channel_id: " << channel_id;

    int code = 0;
    char *err = nullptr;

    g_callback_worker->AddChannelCallback(channel_id, algo_callback);

    // Prepare protobuf's InferenceRequests
    inference::InferenceRequest inference_request;
    inference_request.mutable_data()->set_uri(url);
    inference_request.mutable_data()->set_id(channel_id);
    // Serialize protobuf's InferenceRequests to bytes
    auto inference_request_size = inference_request.ByteSize();
    std::vector<char> inference_request_data(inference_request_size, 0);
    inference_request.SerializeToArray(inference_request_data.data(),
                                       inference_request_size);

    VSSAddStream(g_vss_instance, inference_request_data.data(), inference_request_size, &code, &err);
    if ( code != vss_status_success ) {
        LOG(WARNING) << "vss AddStream code " << code << " != " << vss_status_success << " err: " << err;
        return status_Error;
    }

    LOG(INFO) << "startVidAnalysis success";
    return status_Success;
}

statusCode stopVidAnalysis(const char* json_str) {
    LOG(INFO) << "stopVidAnalysis, params: " << json_str;
    int code = 0;
    char *err = nullptr;

    rapidjson::Document doc;
    if ( doc.Parse(json_str).HasParseError() ) {
        LOG(WARNING) << "stopVidAnalysis parse json error";
        return status_ErrInputParam;
    }
    auto& channel_id = doc["channelId"];

    g_callback_worker->RemoveChannelCallback(channel_id.GetString());

    inference::InferenceRequest inference_request;
    inference_request.mutable_data()->set_id(channel_id.GetString());
    auto inference_request_size = inference_request.ByteSize();
    std::vector<char> inference_request_data(inference_request_size, 0);
    inference_request.SerializeToArray(inference_request_data.data(),
                                       inference_request_size);

    VSSStopStream(g_vss_instance, inference_request_data.data(), inference_request_size, &code, &err);
    if ( code != vss_status_success ) {
        LOG(WARNING) << "vss StopStream code " << code << " != " << vss_status_success << " err: " << err;
        return status_Error;
    }

    LOG(INFO) << "stopVidAnalysis success";
    return status_Success;
}

const char* getVidChannelAnalysisStatus() {
    LOG(INFO) << "getVidChannelAnalysisStatus";
    char* ret_str = new char[1*1024*1024];
    ret_str[0] = '\0';

    if ( g_vss_instance == nullptr ) {
        LOG(WARNING) << "global vss instance is null";
        return ret_str;
    }

    int code = 0;
    char* err = nullptr;
    inference::InferenceResponse inference_response;
    std::vector<char> inference_response_data(4*1024*1024, 0);
    int inference_response_size = 0;

    LOG(INFO) << "starting to VSSGetStatus";
    VSSGetStatus(g_vss_instance, inference_response_data.data(), &inference_response_size, &code, &err);
    if ( code != vss_status_success ) {
        LOG(WARNING) << "vss GetStatus code " << code << " != " << vss_status_success << " err: " << err;
        return ret_str;
    }
    LOG(INFO) << "VSSGetStatus, inference_response_size: " << inference_response_size;
    if ( ! inference_response.ParseFromArray(inference_response_data.data(), inference_response_size) ) {
        LOG(WARNING) << "parse proto error";
    }
    LOG(INFO) << "VSSGetStatus, result: " << inference_response.result();

    strcpy(ret_str, inference_response.result().c_str());

    return ret_str;
}

char* batchPictureAnalysis(const char* json_str) {
    LOG(INFO) << "batchPictureAnalysis (not implemented)";
    // TODO
    char* ret_str = new char[4*1024*1024];
    ret_str[0] = '\0';

    return ret_str;
}