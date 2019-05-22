#include <gtest/gtest.h>
#include "google/protobuf/text_format.h"

#include "core/engines.hpp"

TEST(EnginesTest, Test1) {
  tensord::proto::ModelConfig config;

  google::protobuf::TextFormat::ParseFromString(
      R"(
model:
{
    name: "foo"
    version: 0
    platform: "foo"
}
instance:
{
    model: "foo"
    version: 0
    batchSize: 2
    count:
    {
        count: 1
        kind: GPU
        gpu: 2
    }
})",
      &config);

  tensord::core::Engines engines;
  engines.Set(config);

  std::vector<tensord::core::NetIn<float>> ins = {
      {{"AAA"}, {{0.1, 0.2}}},
      {{"AAA"}, {{0.3, 0.4}}},
  };
  std::vector<tensord::core::NetOut<float>> outs;
  engines.Get("foo", 0)->Predict(ins, &outs);

  EXPECT_EQ(2, outs.size());
}
