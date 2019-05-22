
#include <future>  // NOLINT
#include <iostream>
#include <string>
#include <thread>  // NOLINT

#include <zmq_addon.hpp>
#include "glog/logging.h"

#include "common/time.hpp"
#include "pp.hpp"

static void work(const std::vector<std::pair<size_t, const void *>> &datas,
                 std::vector<std::vector<unsigned char>> *rets) {
  int i = 0;
  for (auto data : datas) {
    rets->at(i).resize(data.first);
    memcpy(&rets->at(i)[0], data.second, data.first);
    i++;
  }
  return;
}

void queue_(std::shared_ptr<zmq::context_t> context,
            std::future<void> exit,
            const std::string &addr_frontend,
            const std::string &addr_backend) {
  batchQueue(std::move(exit), context, "", 4, addr_frontend, addr_backend);
}

void worker_(std::shared_ptr<zmq::context_t> context,
             std::future<void> exit,
             const std::string &addr_,
             void (*work)(const std::vector<std::pair<size_t, const void *>> &,
                          std::vector<std::vector<unsigned char>> *)) {
  worker(std::move(exit), context, addr_, work);
}

void client_(std::shared_ptr<zmq::context_t> context,
             const std::string &, const std::string id, const int &n) {
  auto client = zmq::socket_t(*context, ZMQ_DEALER);
  client.setsockopt(ZMQ_IDENTITY, &id, id.size());
  client.connect("inproc://sdk_cpp_frontend");

  std::vector<float> fs(3);
  std::string body(reinterpret_cast<const char *>(&fs[0]),
                   fs.size() * sizeof(0.1f));

  auto t1 = Time();
  for (int i = 0; i < n; i++) {
    auto t3 = Time();
    zmq::multipart_t msg(body.data(), body.size());
    msg.send(client);

    auto t4 = Time();

    auto t5 = Time();
    zmq::multipart_t ret(client);
    auto t6 = Time();
    LOG(INFO) << "CLIENT: "
              << t6.since_millisecond(t3) << " "
              << t5.since_millisecond(t3) << " "
              << t4.since_millisecond(t3) << " "
              << t5.since_millisecond(t4) << " "
              << t6.since_millisecond(t5) << " "
              << "I:" << i << " -> " << ret.peek(ret.size() - 1)->size();
  }
  auto t2 = Time();
  auto d = t2.since_millisecond(t1);
  LOG(INFO) << "RUN: " << d << " / " << n << " = " << d / n;
}

int main(int, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::INFO);
  google::InstallFailureSignalHandler();
  auto context = std::make_shared<zmq::context_t>(0);
  std::promise<void> e1;
  std::future<void> future1 = e1.get_future();
  std::thread q1(queue_, context, std::move(future1),
                 "inproc://sdk_cpp_frontend", "inproc://sdk_cpp_backend");

  std::vector<std::promise<void>> es;
  std::vector<std::thread> ws;
  for (int i = 0; i < 2; i++) {
    std::promise<void> e;
    std::future<void> future = e.get_future();
    ws.push_back(std::thread(worker_, context, std::move(future),
                             "inproc://sdk_cpp_backend", work));
    es.push_back(std::move(e));
  }

  std::vector<std::thread> cs;
  for (int i = 0; i < 2; i++) {
    cs.push_back(std::thread(client_, context, "inproc://sdk_cpp_frontend",
                             std::to_string(i), 1000));
  }
  for (auto &t : cs) {
    t.join();
  }

  e1.set_value();
  for (auto &e : es) {
    e.set_value();
  }

  q1.join();
  for (auto &t : ws) {
    t.join();
  }

  return 0;
}
