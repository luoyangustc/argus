#ifndef QINIU_CENSOR_CORE_HPP
#define QINIU_CENSOR_CORE_HPP

#include <string>


namespace qc
{
    // error code
    const int ErrTooManyRequests = -1;
    const int ErrEmptyBuf = -2;
    const int ErrTimeout = -3;
    const int ErrInternalServerError = -4;

    struct pic_info_t
    {
        int pid;
        char zip_name[64];
        char bcp_name[64];
        char data_id[64];
        int line;
        char pic_name[64];
    };

    struct metrics_info_t
    {
        // total pictures; total= done + waiting + skip
        int total;
        // skipped pictures
        int skip;
        // pictures in the queue
        int waiting;
        // parsed pictures; done = censor + error
        int done;
        // censor pictures; censor = noraml + pulp + terror + politicion + march + text
        int censor;
        // parse error pictures
        int error;
        // qps of censor pictures in the last 5 seconds
        float last_qps;
        // (1-normal/censor) last 5 seconds
        float last_filter_rate;
        // qps of censor pictures since start up
        float qps;
        // (1-normal/censor) since start up
        float filter_rate;
        // normal pictures
        int normal;
        // pulp pictures
        int pulp;
        // terror pictures
        int terror;
        // politicion pictures
        int politicion;
        // march pictures
        int march;
        // text pictures
        int text;
        // runtime since start up by second
        int runtime;
    };

    enum
    {
        LABEL_NORMAL = 0,
        LABEL_MARCH = 1,
        LABEL_TEXT = 2, 
        LABEL_POLITICIAN = 3,
        LABEL_TERROR = 4,
        LABEL_PULP = 5,
    };

    class Censor
    {
        public:
            Censor();
            virtual ~Censor();
        
        public:
            // 初始化sdk，设置异步eval请求的超时时间
            // timeout: 超时时间
            virtual void init(int timeout);

            // 异步推理请求
            // buf: 图片内容
            // size: 图片内容大小
            // meta: 图片元信息
            // 返回值: 0 - 成功，负数 - 失败
            virtual int eval(const void *buf, const int size, const pic_info_t *meta);

            // 返回当前统计状态
            virtual metrics_info_t metrics();
        
        private:
            int timeout;
    };
}

#endif