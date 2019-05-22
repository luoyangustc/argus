#include <cstring>
#include <cstdio>
#include "gtest/gtest.h"
#include "document.h"
#include "util.hpp"

TEST(MergeDocsTest, Test1) {
    const char* json1 = R"({"a": 1, "b": 2})";
    const char* json2 = R"({"c": 3, "d": 4})";
    rapidjson::Document doc1, doc2;
    EXPECT_EQ(doc1.Parse(json1).HasParseError(), false);
    EXPECT_EQ(doc2.Parse(json2).HasParseError(), false);

    util::merge_docs(doc1, doc2, doc1.GetAllocator());
    EXPECT_EQ(doc1["a"].GetInt(), 1);
    EXPECT_EQ(doc1["b"].GetInt(), 2);
    EXPECT_EQ(doc1["c"].GetInt(), 3);
    EXPECT_EQ(doc1["d"].GetInt(), 4);
}

TEST(MergeDocsTest, Test2) {
    const char* json1 = R"({"a": 1, "b": 2})";
    const char* json2 = R"({"a": 3, "d": 4})";
    rapidjson::Document doc1, doc2;
    EXPECT_EQ(doc1.Parse(json1).HasParseError(), false);
    EXPECT_EQ(doc2.Parse(json2).HasParseError(), false);

    util::merge_docs(doc1, doc2, doc1.GetAllocator());
    EXPECT_EQ(doc1["a"].GetInt(), 3);
    EXPECT_EQ(doc1["b"].GetInt(), 2);
    EXPECT_EQ(doc1["d"].GetInt(), 4);
}

TEST(MergeDocsTest, Test3) {
    const char* json1 = R"({"a": {"b": 1, "c": 2}})";
    const char* json2 = R"({"a": {"c": 3, "d": 4}})";
    rapidjson::Document doc1, doc2;
    EXPECT_EQ(doc1.Parse(json1).HasParseError(), false);
    EXPECT_EQ(doc2.Parse(json2).HasParseError(), false);

    util::merge_docs(doc1, doc2, doc1.GetAllocator());
    EXPECT_EQ(doc1["a"]["b"].GetInt(), 1);
    EXPECT_EQ(doc1["a"]["c"].GetInt(), 3);
    EXPECT_EQ(doc1["a"]["d"].GetInt(), 4);
}

TEST(MergeDocsTest, Test4) {
    const char* json1 = R"({"a": [1, 2]})";
    const char* json2 = R"({"a": [3, 4]})";
    rapidjson::Document doc1, doc2;
    EXPECT_EQ(doc1.Parse(json1).HasParseError(), false);
    EXPECT_EQ(doc2.Parse(json2).HasParseError(), false);

    util::merge_docs(doc1, doc2, doc1.GetAllocator());
    auto arr = doc1["a"].GetArray();
    EXPECT_EQ(arr.Size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
}