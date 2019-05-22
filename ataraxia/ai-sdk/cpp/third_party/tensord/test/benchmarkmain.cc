
#include "benchmark/benchmark.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::LogToStderr();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}

static void BM_StringCreation(::benchmark::State& state) {  // NOLINT
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);
