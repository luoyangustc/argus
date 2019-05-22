#include <map>
#include <string>
#include <thread>
#include "common/Queue.h"
#include "vcs.hpp"

namespace callback {


int StartWaitResultCallback(void *vss_instance);

class CallbackWorker {
public:
    CallbackWorker(void *vss_instance, int worker_num = 5);
    ~CallbackWorker();

    int AsyncWaitAndCallback();
    int AddChannelCallback(const char* channel_id, vid_analysis_cb_func callback);
    int RemoveChannelCallback(const char* channel_id);

private:
    static const int MAX_WORKER_NUM_ = 100;
    void *vss_instance_;
    std::map<std::string, vid_analysis_cb_func> channel_callback_map_;
    std::thread dispath_thread_;
    std::vector<std::thread> callback_threads_;
    int worker_num_;
    char *worker_buffers_[MAX_WORKER_NUM_];
    int worker_buffers_size_[MAX_WORKER_NUM_];
    Queue<int> worker_free_queue_;              // 里面表示目前空闲的 worker index
    Queue<int> worker_signal[MAX_WORKER_NUM_];  // 用于通知 callback worker 线程数据 ready

    void doDispatch();
    void doCallback(int worker_idx);
    
};

void postCallback(const std::string &url, const std::map<std::string, std::string> &fields, int *code, char **err);

}