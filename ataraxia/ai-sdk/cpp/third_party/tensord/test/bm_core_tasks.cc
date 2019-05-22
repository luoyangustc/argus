
#include "benchmark/benchmark.h"
#include "glog/logging.h"

#include "core/tasks.hpp"

static void BM_Tasks(::benchmark::State& state) {  // NOLINT
  using tensord::core::NetIn;
  using tensord::core::NetOut;
  using tensord::core::Task;
  using tensord::core::Tasks;

  Tasks<float> tasks;

  for (auto _ : state) {
    NetIn<float> in({"AAA"}, {{}});
    in.datas[0].resize(3 * 512);
    std::shared_ptr<Task<float>> task1 = tasks.Submit(in);
    std::shared_ptr<Task<float>> task2;
    tasks.TryWait(task2);
    CHECK_EQ(task2->GetIn().datas.size(), 1);
    CHECK_EQ(task2->GetIn().datas[0].size(), 3 * 512);
    NetOut<float> out1({"BBB"}, {{}});
    out1.datas[0].resize(3 * 5);
    task2->SetOut(out1);
    NetOut<float> out2;
    task1->GetOut(&out2);
    CHECK_EQ(out2.datas.size(), 1);
    CHECK_EQ(out2.datas[0].size(), 3 * 5);
  }
}
BENCHMARK(BM_Tasks);
