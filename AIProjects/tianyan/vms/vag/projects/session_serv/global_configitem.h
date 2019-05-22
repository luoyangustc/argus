#pragma once

struct GlobalConfigItem
{
public:
    GlobalConfigItem()
    {
        enable_relay = false;
    }

public:
    bool enable_relay;
};