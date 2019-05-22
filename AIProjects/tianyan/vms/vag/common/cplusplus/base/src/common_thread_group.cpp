#include "common_thread_group.h"
///////////////////////////////////////////////////////////////////////////////////////////

CCommon_Thread_Group::CCommon_Thread_Group()
{
    work_threads_num_ = 4;
    max_task_count_ = 200;
    running_ = false;
}

CCommon_Thread_Group::~CCommon_Thread_Group()
{
    stop();
}

unsigned int CCommon_Thread_Group::set_thread_max_count(int thread_max_count)
{
    work_threads_num_ = thread_max_count;
    return 0;
}

unsigned int CCommon_Thread_Group::set_max_task_count(int task_max_count)
{
    max_task_count_ = task_max_count;
    return 0;
}

unsigned int CCommon_Thread_Group::start()
{
    unsigned int  ulRet = 0;
    do
    {
        running_ = true;
        for( int i=0; i < work_threads_num_; ++i )
        {
            work_threads_.create_thread(boost::bind(&CCommon_Thread_Group::work_main, shared_from_this(), i));
        }
    }while(false);
    return ulRet;
}

unsigned int CCommon_Thread_Group::stop()
{
    unsigned int ulRet = 1;
    do 
    {
        running_ = false;
        work_cond_can_write_.notify_all();
        work_cond_can_read_.notify_all();
        printf("CCommon_Server::stop.\n");
        ulRet = 0;
        return ulRet;
    } while (0);

    return ulRet;
}


bool CCommon_Thread_Group::work_main(unsigned int thread_idx)
{
    printf( "CCommon_Server::work_main--->ThreadIdx(%u).\n", thread_idx );

    while (running_)
    {
        ITaskItem_ptr task_item;
        if ( pop_task(task_item) == 0 )
        {
            //Do the upload work
            task_item->TaskFunc();
            //Do the upload work
        }
    }

    return false;
}

unsigned int   CCommon_Thread_Group::push_task(ITaskItem_ptr  task_in)
{
    boost::mutex::scoped_lock lock(work_mutex_);
    while(work_queue_.size() >= max_task_count_)
    {
        work_cond_can_write_.wait(lock);
    }

    work_queue_.push(task_in);
    work_cond_can_read_.notify_one();
    return 0;
}

unsigned int   CCommon_Thread_Group::pop_task(ITaskItem_ptr&  task_out)
{
    unsigned int  ulRet = 0;
    do 
    {
        boost::mutex::scoped_lock lock(work_mutex_);

        if ( work_queue_.empty() )
        {
            while ( work_queue_.empty() )
            {
                printf("Cloud_Server::pop_cloud_upload_task, enter wait\n");
                work_cond_can_read_.wait(lock);
            }
        }

        if(work_queue_.size() == (max_task_count_ -1))
        {
            work_cond_can_write_.notify_one();
        }

        if( !running_ )
        {
            printf("Cloud_Server::pop_cloud_upload_task, server has stoped!\n");
            break;
        }

        task_out =  work_queue_.front();
        work_queue_.pop();
    } while (0);

    return ulRet;
}

unsigned int   CCommon_Thread_Group::get_unfinished_task_count()
{
    boost::mutex::scoped_lock lock(work_mutex_);
    return work_queue_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////

