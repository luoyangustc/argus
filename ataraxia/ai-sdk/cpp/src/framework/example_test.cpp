#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return r;
}

TEST(Hello, CSAPP1) { EXPECT_EQ(1, 1); }
