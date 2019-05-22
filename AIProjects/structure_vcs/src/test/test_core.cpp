#include <cstring>
#include <cstdio>
#include "gtest/gtest.h"
#include "vcs.hpp"

TEST(InitParamsTest, InvalidFormat) {
    const char* invalid_json1 = "asdfg";

    EXPECT_EQ(
        initAnalysisModule(invalid_json1, strlen(invalid_json1)),
        status_ErrInputParam);
}

TEST(InitParamsTest, ValidFormat) {
    const char* valid_json1 = "{}";

    statusCode code = initAnalysisModule(valid_json1, strlen(valid_json1));
    printf("code: %d\n", code);
    EXPECT_EQ(code, status_Success);
}

TEST(GetStatusTest, Normal) {
    const char* valid_json1 = "{}";

    statusCode code = initAnalysisModule(valid_json1, strlen(valid_json1));
    printf("start to getVidChannelAnalysisStatus\n");
    const char* status_str = getVidChannelAnalysisStatus();
    EXPECT_GT(strlen(status_str), 0);
}