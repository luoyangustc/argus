#include <gtest/gtest.h>
#include "google/protobuf/text_format.h"

#include "core/engine.hpp"

TEST(CreateNetsTest, Test1) {
  tensord::proto::Model modelConfig;
  tensord::proto::Instance instanceConfig;

  google::protobuf::TextFormat::ParseFromString(
      R"(batchSize: 2
count:
{
    count: 2
    batchSize: 4
    kind: GPU
    gpu: 0
    gpu: 1
}
count:
{
    count: 2
    batchSize: 4
    kind: CPU
}
count:
{
    count: 1
    kind: GPU
    gpu: 2
})",
      &instanceConfig);

  std::vector<std::shared_ptr<tensord::core::Net<float>>> nets;
  std::vector<int> batch_sizes;
  tensord::core::CreateNets<float>(
      [](tensord::proto::Model,
         tensord::proto::Instance::Kind,
         int,
         int) -> std::shared_ptr<tensord::core::Net<float>> {
        auto ptr = std::make_shared<tensord::core::NetFoo>();
        return std::static_pointer_cast<tensord::core::Net<float>>(ptr);
      },
      modelConfig,
      instanceConfig,
      &nets,
      &batch_sizes);

  EXPECT_EQ(7, nets.size());
  EXPECT_EQ(7, batch_sizes.size());
  EXPECT_EQ(4, batch_sizes[0]);
  EXPECT_EQ(4, batch_sizes[1]);
  EXPECT_EQ(4, batch_sizes[2]);
  EXPECT_EQ(4, batch_sizes[3]);
  EXPECT_EQ(4, batch_sizes[4]);
  EXPECT_EQ(4, batch_sizes[5]);
  EXPECT_EQ(2, batch_sizes[6]);
}

TEST(EngineTest, Test1) {
  tensord::proto::Model modelConfig;
  tensord::proto::Instance instanceConfig;

  google::protobuf::TextFormat::ParseFromString(
      R"(batchSize: 2
count:
{
    count: 1
    kind: GPU
    gpu: 2
})",
      &instanceConfig);

  tensord::core::Engine<float> engine;
  engine.Setup(
      [](tensord::proto::Model,
         tensord::proto::Instance::Kind,
         int,
         int) -> std::shared_ptr<tensord::core::Net<float>> {
        auto ptr = std::make_shared<tensord::core::NetFoo>();
        return std::static_pointer_cast<tensord::core::Net<float>>(ptr);
      },
      modelConfig,
      instanceConfig);

  std::vector<tensord::core::NetIn<float>> ins = {
      {{"AAA"}, {{0.1, 0.2}}},
      {{"AAA"}, {{0.3, 0.4}}},
  };
  std::vector<tensord::core::NetOut<float>> outs;
  engine.Predict(ins, &outs);

  EXPECT_EQ(2, outs.size());
}
