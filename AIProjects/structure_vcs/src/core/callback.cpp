#include <cstdio>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <map>
#include <pthread.h>
#include "restclient-cpp/restclient.h"
#include "vss.hpp"
#include "vcs.hpp"
#include "proto/inference.pb.h"
#include "common/log.hpp"
#include "util.hpp"
#include "callback.hpp"

namespace callback {

CallbackWorker::CallbackWorker(void *vss_instance, int worker_num):
    vss_instance_(vss_instance),
    worker_num_(worker_num) {

    if (worker_num_ > MAX_WORKER_NUM_) {
        worker_num_ = MAX_WORKER_NUM_;
    }
    for ( int i = 0; i < worker_num_; i ++ ) {
        worker_buffers_[i] = new char[responses_max_buffer_size];
    }
}

CallbackWorker::~CallbackWorker() {
    LOG(INFO) << "CallbackWorker::~CallbackWorker";
    for ( int i = 0; i < worker_num_; i ++ ) {
        delete[] worker_buffers_[i];
    }
    // TODO, notify threads to quit

    return;
}

int CallbackWorker::AddChannelCallback(const char* channel_id, vid_analysis_cb_func callback) {
    LOG(INFO) << "CallbackWorker::AddChannelCallback, channel: " << channel_id;
    channel_callback_map_[std::string(channel_id)] = callback;
    return 0;
}

int CallbackWorker::RemoveChannelCallback(const char* channel_id) {
    LOG(INFO) << "CallbackWorker::RemoveChannelCallback, channel: " << channel_id;
    channel_callback_map_.erase(std::string(channel_id));
    return 0;
}

int CallbackWorker::AsyncWaitAndCallback() {
    LOG(INFO) << "CallbackWorker::AsyncWaitAndCallback";

#ifndef USE_MOCK_VSS
    dispath_thread_ = std::thread(&CallbackWorker::doDispatch, this);
    for ( int i = 0; i < worker_num_; i ++ ) {
        callback_threads_.push_back(std::thread(&CallbackWorker::doCallback, this, i));
    }
#endif

    return 0;
}

void CallbackWorker::doDispatch() {
    int code = 0;
    char *err = nullptr;

    while (true) {
        auto free_worker_idx = worker_free_queue_.pop();
        VSSProcess(vss_instance_, worker_buffers_[free_worker_idx], &worker_buffers_size_[free_worker_idx], &code, &err);
        if (code != vss_status_success)
        {
            LOG(WARNING) << "VSSProcess code " << code << " != " << vss_status_success << " err: " << err;
            goto SIGNAL;
        }
SIGNAL:
        worker_signal[free_worker_idx].push(1);
    }

}

void CallbackWorker::doCallback(int worker_idx) {
    LOG(INFO) << "CallbackWorker::doProcessCallback, worker " << worker_idx;

    int code = 0;
    char* err = nullptr;
    int inference_response_size = 0;
    inference::InferenceResponse inference_response;
    std::map<std::string, std::string> fields;

    while (true) {
        // 表示当前 worker 处于空闲状态
        worker_free_queue_.push(worker_idx);
        // 等待主线程准备好数据
        worker_signal[worker_idx].pop();

        inference_response.Clear();
        auto succ = inference_response.ParseFromArray(worker_buffers_[worker_idx], worker_buffers_size_[worker_idx]);
        if ( ! succ ) {
            LOG(WARNING) << "worker " << worker_idx << " parse buffer failed";
            continue;
        }
        for (int i = 0; i < inference_response.infos_size(); i ++) {
            auto info = inference_response.infos(i);
            LOG(INFO) << "worker: "<< worker_idx << " channel: " << info.channel().c_str() << " json_len: " << info.json().length();
            auto channel = info.channel();
            auto search = channel_callback_map_.find(channel);
            if ( search != channel_callback_map_.end() ) {
                auto cb = search->second;
                cb(info.json().c_str(), info.channel().c_str());
            }
        }
   }
}

void postCallback(const std::string &url, const std::map<std::string, std::string> &fields, int *code, char **err) {
    std::stringstream ss;
    for (std::map<std::string,std::string>::const_iterator it = fields.begin(); it != fields.end(); ++it) {
        ss << util::urlencode(it->first) << "=" << util::urlencode(it->second) << "&";
    }
    RestClient::Response r = RestClient::post(url, "application/x-www-form-urlencoded", ss.str());
    if ( r.code != 200 ) {
        *code = 1;
        *err = const_cast<char *>("code != 200");
        LOG(WARNING) << "post url code != 200, url " << url;
    }
    *code = 0;

    return;
}

}
