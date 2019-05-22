#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <limits>
#include <climits>


template<typename T>
class Queue {
  public:
    Queue() : interrupt_{false} {}

    Queue(unsigned int max_size) : interrupt_{false}, max_size_{max_size} {}

    void push(T item) {
        {
            std::unique_lock<std::mutex> lock{lock_};
            cond_full_.wait(lock, [&](){ unsigned long size = queue_.size();
                return (size < max_size_) || interrupt_;});
//        }
//
//        {
//            std::lock_guard<std::mutex> lock(lock_);
            queue_.push(std::move(item));
        }
        cond_.notify_one();
    }

    void force_push(T item) {
        {
            std::unique_lock<std::mutex> lock{lock_};
            unsigned long size = queue_.size();
            if (size >= max_size_) {
                queue_.pop();
            }

            queue_.push(std::move(item));
        }
        cond_.notify_one();
    }

    T pop() {
        static auto int_return = T{};
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
        if (interrupt_) {
            return std::move(int_return);
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cond_full_.notify_one();
        return item;
    }

    T pop(int timeout) {
        static auto int_return = T{};
        const std::chrono::milliseconds t(timeout);
        std::unique_lock<std::mutex> lock{lock_};
        if (cond_.wait_for(lock, t, [&](){return !queue_.empty() || interrupt_;}) == false) {
            throw std::runtime_error("Queue pop timeout error!");
        }
        if (interrupt_) {
            return std::move(int_return);
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cond_full_.notify_one();
        return item;
    }

    const T& peek() {
        static auto int_return = T{};
        std::unique_lock<std::mutex> lock{lock_};
        cond_.wait(lock, [&](){return !queue_.empty() || interrupt_;});
        if (interrupt_) {
            return std::move(int_return);
        }
        return queue_.front();
    }

    bool empty() const {
        return queue_.empty();
    }

    typename std::queue<T>::size_type size() const {
        return queue_.size();
    }

    void cancel_pops() {
        std::lock_guard<std::mutex> lock(lock_);
        interrupt_ = true;
        cond_.notify_all();
        cond_full_.notify_all();
    }

    void clear() {
        std::unique_lock<std::mutex> lock{lock_};
        std::queue<T> empty;
        std::swap(queue_, empty);
    }

  private:
    std::queue<T> queue_;
    std::mutex lock_;
    std::condition_variable cond_;
    std::condition_variable cond_full_;
    std::atomic<bool> interrupt_;
    unsigned int max_size_ = UINT_MAX;
};
