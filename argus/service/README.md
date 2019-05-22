
## 代码组织

```
/service
├── /sample                 // APP示例
│   ├── /argus_image        // 同步图片应用示例
│   └── /argus_video        // 视频应用示例
├── /scenario               // 场景模块
│   ├── /image_sync         // 同步图片服务
│   ├── /video              // 视频服务
│   └── main.go             // 场景统一main函数模板
└── /service
    ├── /image              // 图片相关服务
    │   ├── /pulp           // 剑皇服务
    │   │   └── /image_sync // 注册同步图片场景
    │   │       ├── /test   // 集成测试案例
    │   │       └── /benchmark  // 性能测试案例
    │   └── /foo
    └── /video              // 视频相关服务
```