#include <gtest/gtest.h>

#include "core/tasks.hpp"

TEST(TasksTest, Test1) {
  tensord::core::Tasks<float> tasks;

  tensord::core::NetIn<float> in;
  tensord::core::NetOut<float> out1, out2;
  std::shared_ptr<tensord::core::Task<float>> task1, task2;

  EXPECT_FALSE(tasks.TryWait(task2));

  in.names = {"AAA"}, in.datas = {{0.1, 0.2}};
  task1 = tasks.Submit(in);

  EXPECT_TRUE(tasks.TryWait(task2));
  EXPECT_EQ(1, task2->GetIn().names.size());
  EXPECT_EQ("AAA", task2->GetIn().names[0]);
  EXPECT_EQ(2, task2->GetIn().datas[0].size());
  EXPECT_FLOAT_EQ(0.1, task2->GetIn().datas[0][0]);
  EXPECT_FLOAT_EQ(0.2, task2->GetIn().datas[0][1]);

  out1.names = {"BBB", "CCC"}, out1.datas = {{1.1, 1.2}, {2.1, 2.2}};
  task2->SetOut(out1);
  task1->GetOut(&out2);
  EXPECT_EQ(2, out2.names.size());
  EXPECT_EQ("BBB", out2.names[0]);
  EXPECT_FLOAT_EQ(1.1, out2.datas[0][0]);
  EXPECT_FLOAT_EQ(1.2, out2.datas[0][1]);
  EXPECT_EQ("CCC", out2.names[1]);
  EXPECT_FLOAT_EQ(2.1, out2.datas[1][0]);
  EXPECT_FLOAT_EQ(2.2, out2.datas[1][1]);
}
