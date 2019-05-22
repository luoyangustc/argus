#include "Util.hpp"

namespace Shadow
{

__half float2half(float f)
{
    // 先进型强制转换，x与原始f拥有相同的比特位
    uint32_t x = bitwise_cast<uint32_t, float>(f);
    // 将符号位移除，成为无符号u
    uint32_t u = (x & 0x7fffffff);

    // Get rid of +NaN/-NaN case first.
    // float的指数位8位全部为1，直接判定为非数字
    if (u > 0x7f800000)
        return bitwise_cast<__half, uint16_t>(uint16_t(0x7fff));

    // 提取符号位
    uint16_t sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    // u > 65520 超出半精度表示上限，指数位全1，底数位全0表示超出上限
    if (u > 0x477fefff)
        return bitwise_cast<__half, uint16_t>(sign | uint16_t(0x7c00));
    // u < 2^-14 超出半精度表示下限，输出0
    if (u < 0x33000001)
        return bitwise_cast<__half, uint16_t>(sign | uint16_t(0x0000));

    // 提取指数位,float后23位位底数位，因此先右移23位，在取8位指数位
    uint32_t exponent = ((u >> 23) & 0xff);
    // 提取底数位8，取后23位
    uint32_t mantissa = (u & 0x7fffff);

    //偏移值>=13
    uint32_t shift;
    // 指数位>112
    if (exponent > 0x70)
    {
        shift = 13;
        exponent -= 0x70;
    }
    else
    {
        // 偏移 = 126-指数位 >= 14
        shift = 0x7e - exponent;
        exponent = 0;
        // 原底数只有23位，本质是24位，最高位默认是1
        mantissa |= 0x800000;
    }

    // lsb=2^13, lsb_s1=2^12, lsb_s1=2^13-1（12位全1）
    uint32_t lsb = (1 << shift);
    uint32_t lsb_s1 = (lsb >> 1);
    uint32_t lsb_m1 = (lsb - 1);

    // Round to nearest even.
    // 暂时保留底数的后12位
    uint32_t remainder = (mantissa & lsb_m1);
    // 将底数偏移13位得到半精度所需的10位底数
    mantissa >>= shift;
    // 对后面12位的精度进行四舍五入（底数位奇数时，remainder==1/2时即进位）
    if ((remainder > lsb_s1) || ((remainder == lsb_s1) && (mantissa & 0x1)))
    {
        ++mantissa;
        // 如果底数位自增后为0，说明产生进位，指数+1
        if (!(mantissa & 0x3ff))
        {
            ++exponent;
            mantissa = 0;
        }
    }
    // 组合符号位，指数位，底数位
    return bitwise_cast<__half, uint16_t>(sign | uint16_t(exponent << 10) | uint16_t(mantissa));
}

float half2float(__half h)
{
    uint16_t x = bitwise_cast<uint16_t, __half>(h);
    uint32_t sign = ((x >> 15) & 1);
    uint32_t exponent = ((x >> 10) & 0x1f);
    uint32_t mantissa = (static_cast<uint32_t>(x & 0x3ff) << 13);

    if (exponent == 0x1f)
    { /* NaN or Inf */
        if (mantissa != 0)
        { // NaN
            sign = 0;
            mantissa = 0x7fffff;
        }
        else // Inf
            mantissa = 0;
        exponent = 0xff;
    }
    else if (!exponent)
    { /* Denorm or Zero */
        if (mantissa)
        {
            unsigned int msb;
            exponent = 0x71;
            do
            {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    }
    else
        exponent += 0x70;
    return bitwise_cast<float, uint32_t>((sign << 31) | (exponent << 23) | mantissa);
}

size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

void *copyToDevice(const void *data, size_t count)
{
    void *deviceData;
    CHECK(cudaMalloc(&deviceData, count));
    CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
    return deviceData;
}

void convertAndCopyToDevice(void *&deviceWeights, const Weights &weights, DataType mDataType)
{
    if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
    {
        size_t size = weights.count * (mDataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
        void *buffer = malloc(size);
        for (int64_t v = 0; v < weights.count; ++v)
            if (mDataType == DataType::kFLOAT)
                static_cast<float *>(buffer)[v] = Shadow::half2float(static_cast<const __half *>(weights.values)[v]);
            else
                static_cast<__half *>(buffer)[v] = float2half(static_cast<const float *>(weights.values)[v]);

        deviceWeights = copyToDevice(buffer, size);
        free(buffer);
    }
    else
        deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
}

void convertAndCopyToBuffer(char *&buffer, const Weights &weights, DataType mDataType)
{
    if (weights.type != mDataType)
        for (int64_t v = 0; v < weights.count; ++v)
            if (mDataType == DataType::kFLOAT)
                reinterpret_cast<float *>(buffer)[v] = Shadow::half2float(static_cast<const __half *>(weights.values)[v]);
            else
                reinterpret_cast<__half *>(buffer)[v] = float2half(static_cast<const float *>(weights.values)[v]);
    else
        memcpy(buffer, weights.values, weights.count * type2size(mDataType));
    buffer += weights.count * type2size(mDataType);
}

void deserializeToDevice(const char *&hostBuffer, void *&deviceWeights, size_t size)
{
    deviceWeights = copyToDevice(hostBuffer, size);
    hostBuffer += size;
}

} // namespace Shadow
