#ifndef TRON_FRAMEWORK_PP_HPP  // NOLINT
#define TRON_FRAMEWORK_PP_HPP

#include <stdint.h>
#include <chrono>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <zmq.hpp>
#include <zmq_addon.hpp>
#include "glog/logging.h"

#include "zhelpers.hpp"

#include "common/time.hpp"

#define HEARTBEAT_LIVENESS 3     //  3-5 is reasonable
#define HEARTBEAT_INTERVAL 1000  //  msecs
#define INTERVAL_INIT 1000       //  Initial reconnect
#define INTERVAL_MAX 32000       //  After exponential backoff

//  This defines one active worker in our worker queue

typedef struct {
  std::string identity;  //  Address of worker
  int64_t expiry;        //  Expires at this time
} worker_t;

//  Insert worker at end of queue, reset expiry
//  Worker must not already be in queue
static void s_worker_append(std::vector<worker_t> *queue,
                            const std::string &identity) {
  bool found = false;
  for (auto it = queue->begin(); it < queue->end(); it++) {
    if (it->identity.compare(identity) == 0) {
      LOG(INFO) << "E: duplicate worker identity " << identity;
      found = true;
      break;
    }
  }
  if (!found) {
    worker_t worker;
    worker.identity = identity;
    worker.expiry = s_clock() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS;
    queue->push_back(worker);
  }
}

//  Remove worker from queue, if present
static void s_worker_delete(std::vector<worker_t> *queue,
                            const std::string &identity) {
  for (auto it = queue->begin(); it < queue->end(); it++) {
    if (it->identity.compare(identity) == 0) {
      it = queue->erase(it);
      break;
    }
  }
}

//  Reset worker expiry, worker must be present
static void s_worker_refresh(std::vector<worker_t> *queue,
                             const std::string &identity) {
  bool found = false;
  for (auto it = queue->begin(); it < queue->end(); it++) {
    if (it->identity.compare(identity) == 0) {
      it->expiry = s_clock() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS;
      found = true;
      break;
    }
  }
  if (!found) {
    LOG(WARNING) << "E: worker " << identity << " not ready";
  }
}

//  Pop next available worker off queue, return identity
static std::string s_worker_dequeue(std::vector<worker_t> *queue) {
  assert(queue->size());
  std::string identity = queue->at(0).identity;
  queue->erase(queue->begin());
  return identity;
}

//  Look for & kill expired workers
static void s_queue_purge(std::vector<worker_t> *queue) {
  int64_t clock = s_clock();
  for (auto it = queue->begin(); it < queue->end(); it++) {
    if (clock > it->expiry) {
      it = queue->erase(it) - 1;
    }
  }
}

static std::unique_ptr<zmq::socket_t> s_worker_socket(zmq::context_t &context,  // NOLINT
                                                      std::string const &addr_,
                                                      std::string *identity) {
  auto worker = std::make_unique<zmq::socket_t>(context, ZMQ_DEALER);

  //  Set random identity to make tracing easier
  *identity = s_set_id(*worker);
  worker->connect(addr_);

  //  Configure socket to not wait at close time
  int linger = 0;
  worker->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

  //  Tell queue we're ready for work
  LOG(INFO) << "I: (" << *identity << ") worker ready";
  s_send(*worker, "READY");

  return worker;
}

static inline void worker(
    std::future<void> exit,
    std::shared_ptr<zmq::context_t> context,
    std::string const &addr_,
    std::function<void(
        const std::vector<std::pair<size_t, const void *>> &,
        std::vector<std::vector<unsigned char>> *)>
        work) {
  s_version_assert(4, 0);
  srandom((unsigned)time(NULL));

  std::string identity;
  auto worker = s_worker_socket(*context, addr_, &identity);

  //  If liveness hits zero, queue is considered disconnected
  size_t liveness = HEARTBEAT_LIVENESS;
  size_t interval = INTERVAL_INIT;

  //  Send out heartbeats at regular intervals
  int64_t heartbeat_at = s_clock() + HEARTBEAT_INTERVAL;

  // int cycles = 0;
  while (exit.wait_for(std::chrono::microseconds(1)) ==
         std::future_status::timeout) {
    zmq::pollitem_t items[] = {{*worker, 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, HEARTBEAT_INTERVAL);

    if (items[0].revents & ZMQ_POLLIN) {
      //  Get message
      //  - 3-part envelope + content -> request
      //  - 1-part "HEARTBEAT" -> heartbeat

      std::vector<zmq::message_t> datas;
      std::vector<zmq::multipart_t> rets;
      while (1) {
        zmq::multipart_t msg;
        if (!msg.recv(*worker, ZMQ_DONTWAIT)) {
          if (datas.size()) {
            auto t1 = Time();
            auto bodys = std::vector<std::vector<unsigned char>>(datas.size());
            std::vector<std::pair<size_t, const void *>> buf(datas.size());
            for (std::size_t i = 0; i < datas.size(); i++) {
              buf[i] = {datas[i].size(), datas[i].data()};
            }
            work(buf, &bodys);
            auto t2 = Time();
            zmq::multipart_t ret;
            for (size_t i = 0; i < bodys.size(); i++) {
              ret.append(std::move(rets[i]));
              ret.addmem(&bodys[i][0], bodys[i].size());
            }
            ret.send(*worker);
            // for (int i = 0; i < bodys.size(); i++) {
            //   auto ret = std::move(rets[i]);
            //   ret.addmem(&bodys[i][0], bodys[i].size());
            //   ret.send(*worker);
            // }
            liveness = HEARTBEAT_LIVENESS;
            auto t3 = Time();
            LOG(INFO) << "WORKER: "
                      << t2.since_millisecond(t1) << " "
                      << t3.since_millisecond(t1)
                      << " / " << datas.size() << " = "
                      << t2.since_millisecond(t1) / datas.size() << " "
                      << t3.since_millisecond(t1) / datas.size();
          }
          break;
        }

        if (msg.size() > 0 && msg.size() % 2 == 0) {
          for (std::size_t i = 0; i < msg.size(); i += 2) {
            zmq::multipart_t ret;
            ret.addmem(msg.peek(i)->data(), msg.peek(i)->size());
            zmq::message_t data(msg.peek(i + 1)->data(),
                                msg.peek(i + 1)->size());
            datas.push_back(std::move(data));
            rets.push_back(std::move(ret));
          }
          // zmq::multipart_t ret;
          // for (int i = 1; i < msg.size() - 1; i++) {
          //   ret.addmem(msg.peek(i)->data(), msg.peek(i)->size());
          // }
          // zmq::message_t data(msg.peek(msg.size() - 1)->data(),
          //                     msg.peek(msg.size() - 1)->size());
          // datas.push_back(std::move(data));
          // rets.push_back(std::move(ret));

        } else {
          if (msg.size() == 1 &&
              strcmp(msg.peekstr(msg.size() - 1).c_str(), "HEARTBEAT") == 0) {
            liveness = HEARTBEAT_LIVENESS;
          } else {
            LOG(ERROR) << "E: ( " << identity << ") invalid message."
                       << " size: " << identity,
                msg.size();
            // msg.dump();
          }
        }
      }
      interval = INTERVAL_INIT;
    } else if (--liveness == 0) {
      LOG(ERROR) << "W: (" << identity
                 << ") heartbeat failure, can't reach queue";
      LOG(ERROR) << "W: (" << identity << ") reconnecting in "
                 << interval << " msec...";
      s_sleep(interval);

      if (interval < INTERVAL_MAX) {
        interval *= 2;
      }
      worker = s_worker_socket(*context, addr_, &identity);
      liveness = HEARTBEAT_LIVENESS;
    }

    //  Send heartbeat to queue if it's time
    if (s_clock() > heartbeat_at) {
      heartbeat_at = s_clock() + HEARTBEAT_INTERVAL;
      s_send(*worker, "HEARTBEAT");
    }
  }
  LOG(INFO) << "end of worker " << identity;
  return;
}

static inline void batchQueue(std::future<void> exit,
                              std::shared_ptr<zmq::context_t> context,
                              const std::string name, const int batch_size,
                              const std::string &addr_frontend,
                              const std::string &addr_backend) {
  s_version_assert(4, 0);

  //  Prepare our context and sockets
  zmq::socket_t frontend(*context, ZMQ_ROUTER);
  // zmq::socket_t frontend2(context, ZMQ_PULL);
  zmq::socket_t backend(*context, ZMQ_ROUTER);
  frontend.bind(addr_frontend);  //  For clients
  backend.bind(addr_backend);    //  For workers

  //  Queue of available workers
  std::vector<worker_t> queue;
  std::vector<zmq::multipart_t> messageQueue;

  //  Send out heartbeats at regular intervals
  int64_t heartbeat_at = s_clock() + HEARTBEAT_INTERVAL;

  auto t1 = Time();
  auto t3 = Time();

  while (exit.wait_for(std::chrono::microseconds(1)) ==
         std::future_status::timeout) {
    zmq::pollitem_t items[] = {{backend, 0, ZMQ_POLLIN, 0},
                               {frontend, 0, ZMQ_POLLIN, 0}};
    //  Poll frontend only if we have available workers
    zmq::poll(items, 2, HEARTBEAT_INTERVAL);

    //  Handle worker activity on backend
    if (items[0].revents & ZMQ_POLLIN) {
      while (1) {
        zmq::multipart_t msg;
        if (!msg.recv(backend, ZMQ_DONTWAIT)) {
          break;
        }
        std::string identity(msg.popstr());

        //  Return reply to client if it's not a control message
        if (msg.size() == 1) {
          if (strcmp(msg.peekstr(0).c_str(), "READY") == 0) {
            s_worker_delete(&queue, identity);
            s_worker_append(&queue, identity);
          } else {
            if (strcmp(msg.peekstr(0).c_str(), "HEARTBEAT") == 0) {
              s_worker_refresh(&queue, identity);
            } else {
              LOG(WARNING) << "E: invalid message from " << identity;
              // msg.dump();
            }
          }
        } else {
          auto t4 = Time();

          LOG(INFO) << msg.size();

          // msg.popstr();
          for (size_t i = 0; i < msg.size(); i += 2) {
            zmq::multipart_t msg2;
            msg2.addmem(msg.peek(i)->data(), msg.peek(i)->size());
            msg2.addmem(msg.peek(i + 1)->data(), msg.peek(i + 1)->size());
            msg2.send(frontend);
          }
          // msg.send(frontend);

          auto t2 = Time();
          LOG(INFO) << "PROXY2: " << t2.since_millisecond(t4)
                    << " " << t4.since_millisecond(t3);

          s_worker_delete(&queue, identity);
          s_worker_append(&queue, identity);
        }
      }
      if (queue.size() && messageQueue.size()) {
        auto n = messageQueue.size();
        auto identity = std::string(s_worker_dequeue(&queue));
        zmq::multipart_t msg2;
        for (int i = 0; i < batch_size && messageQueue.size(); i++) {
          msg2.append(std::move(messageQueue[0]));
          messageQueue.erase(messageQueue.begin());
          // auto msg = std::move(messageQueue[0]);
          // messageQueue.erase(messageQueue.begin());
          // msg.pushstr("");
          // msg.pushstr(identity);
          // msg.send(backend);
        }
        msg2.pushstr(identity);
        msg2.send(backend);
        LOG(INFO) << name << " " << n << " --> " << messageQueue.size();
      }
    }
    if (items[1].revents & ZMQ_POLLIN) {
      //  Now get next client request, route to next worker
      t1 = Time();

      zmq::multipart_t msg(frontend);
      messageQueue.push_back(std::move(msg));
      LOG(INFO) << name << " " << messageQueue.size();

      if (queue.size()) {
        auto n = messageQueue.size();
        std::string identity = std::string(s_worker_dequeue(&queue));
        zmq::multipart_t msg2;
        for (int i = 0; i < batch_size && messageQueue.size(); i++) {
          msg2.append(std::move(messageQueue[0]));
          messageQueue.erase(messageQueue.begin());
        }
        msg2.pushstr(identity);
        msg2.send(backend);
        LOG(INFO) << name << " " << n << " --> " << messageQueue.size();
      }
    }

    //  Send heartbeats to idle workers if it's time
    if (s_clock() > heartbeat_at) {
      for (std::vector<worker_t>::iterator it = queue.begin(); it < queue.end();
           it++) {
        zmq::multipart_t msg("HEARTBEAT");
        msg.pushstr(it->identity);
        msg.send(backend);
      }
      heartbeat_at = s_clock() + HEARTBEAT_INTERVAL;
    }
    s_queue_purge(&queue);
  }
  LOG(INFO) << "end of queue";

  //  We never exit the main loop
  //  But pretend to do the right shutdown anyhow
  queue.clear();
  return;
}

#endif  // TRON_FRAMEWORK_PP_HPP NOLINT