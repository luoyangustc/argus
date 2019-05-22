#ifndef __COMMON_THREAD_GROUP__
#define __COMMON_THREAD_GROUP__

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>

using namespace std;


class ITaskItem
{
public:
    ITaskItem(){}
    virtual ~ITaskItem(){}
    virtual unsigned int TaskFunc() = 0;
};

typedef boost::shared_ptr<ITaskItem> ITaskItem_ptr;


class CCommon_Thread_Group:public boost::enable_shared_from_this<CCommon_Thread_Group>
{
public:
    CCommon_Thread_Group();
    ~CCommon_Thread_Group();
    unsigned int start();
    unsigned int set_thread_max_count(int thread_max_count);
    unsigned int set_max_task_count(int task_max_count);
    unsigned int stop();
    unsigned int   push_task(ITaskItem_ptr  task_in);
    unsigned int   get_unfinished_task_count();
	
private:
    bool work_main(unsigned int thread_idx);
    unsigned int   pop_task(ITaskItem_ptr&  task_out);
	
	
private:
    boost::mutex                        work_mutex_;
    boost::condition                    work_cond_can_read_;
    boost::condition                    work_cond_can_write_;
    std::queue<ITaskItem_ptr>           work_queue_;
    unsigned int                        work_threads_num_;
    boost::thread_group                 work_threads_;
    bool                                running_;
    int                                 max_task_count_;
};

typedef boost::shared_ptr<CCommon_Thread_Group> CCommon_Thread_Group_ptr;
#endif