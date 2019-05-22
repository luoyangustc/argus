#include "./utils.hpp"
#include "gtest/gtest.h"

namespace tron {
namespace terror_mixup {

TEST(TerrorMixup, NetworkShape) {
  vector<float> data(400);
  NetworkShape("test1",
               {-1, 10, 10}, NetworkShapeType::Normal)
      .assert_shape_match(data, 4);
  EXPECT_EQ(NetworkShape("test1", {-1, 10, 10}, NetworkShapeType::Normal)
                .single_batch_size(),
            100);

  vector<float> data2(100);
  NetworkShape("test1",
               {-1, 10, 10},
               NetworkShapeType::BatchSizeUnrelated)
      .assert_shape_match(data2, 4);
  vector<float> data3(400);
  NetworkShape("test1",
               {-1, 10, 10},
               NetworkShapeType::Normal)
      .assert_shape_match(data3, 4);
  EXPECT_EQ(NetworkShape("test1",
                         {-1, 10, 10},
                         NetworkShapeType::Normal)
                .single_batch_size(),
            100);
}

TEST(TerrorMixup, Split) {
  {
    vector<string> r = {"hello", "world"};
    EXPECT_EQ(split_str("hello world", " "), r);
    EXPECT_EQ(split_str("hello  world", " "), r);
  }
  {
    vector<string> r = {"hello", "world", "good", "2"};
    EXPECT_EQ(split_str("hello#@world@good#2", "@#"), r);
  }
  {
    vector<string> r = {"a"};
    EXPECT_EQ(split_str("a", ","), r);
  }
  {
    vector<string> r = {""};
    EXPECT_EQ(split_str("", ","), r);
  }
}

TEST(TerrorMixup, ParseCSV) {
  const string csv = R"(

index,class,threshold,clsNeed
1,islamic flag,0.15,Predet
2,isis flag,0.15,Predet
3,tibetan flag,0.15,Predet

4,knives,0.15,Predet
5,guns,0.15,Predet
6,bloodiness,0.6,yes_0


  )";
  vector<vector<string>> r = {
      {"index", "class", "threshold", "clsNeed"},
      {"1", "islamic flag", "0.15", "Predet"},
      {"2", "isis flag", "0.15", "Predet"},
      {"3", "tibetan flag", "0.15", "Predet"},
      {"4", "knives", "0.15", "Predet"},
      {"5", "guns", "0.15", "Predet"},
      {"6", "bloodiness", "0.6", "yes_0"}};
  auto r2 = csv_parse(csv);
  EXPECT_EQ(r, r2);
}

TEST(TerrorMixup, ReadFile) {
  auto a = read_bin_file_to_string("/etc/hosts");
  EXPECT_NE(a, "");
  ASSERT_THROW(read_bin_file_to_string("/etc/xxxx"), std::invalid_argument);
}

}  // namespace terror_mixup
}  // namespace tron
