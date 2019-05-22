
## 性能测试报告 censor-unknow_product_version 20181108200813 unknow_deploy_kind

> 测试工具版本：v0.1， 运行参数 bench.py run -c ./censor/image_sync/benchmark/censor/benchmark_config.yaml -d 10s

|测试名称|测试数据集|并发|检查|QPS|响应时间(95%)|(90%)|max|min|med|arg|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|censor 全部op pulp\|terror\|politician|all.tsv|1|100.00%|1.799982/s|1.06s|890.8ms|1.61s|190.04ms|439.2ms|536.28ms|
|censor 全部op pulp\|terror\|politician|all.tsv|10|100.00%|2.599972/s|3.95s|3.65s|4.29s|1.6s|2.92s|2.88s|
|censor 单op pulp|all.tsv|1|100.00%|4.599953/s|351.68ms|338.29ms|425.97ms|93.2ms|209.05ms|215.48ms|
|censor 单op pulp|all.tsv|10|100.00%|7.599951/s|1.7s|1.64s|1.77s|106.31ms|1.17s|1.18s|
|censor 单op pulp|pulp-normal.tsv|1|100.00%|3.899959/s|485.05ms|429.15ms|541.06ms|73.59ms|207.56ms|246.34ms|
|censor 单op pulp|pulp-normal.tsv|10|100.00%|5.399974/s|2.31s|2.17s|2.52s|111.96ms|1.76s|1.7s|
|censor 单op pulp|pulp.tsv|1|100.00%|9.599897/s|138.38ms|119.18ms|152.6ms|85.57ms|98.36ms|103.29ms|
|censor 单op pulp|pulp.tsv|10|100.00%|11.799877/s|898.12ms|884.13ms|918.16ms|92.91ms|837.79ms|810.73ms|
|censor 单op terror|all.tsv|1|100.00%|2.999968/s|521.79ms|492.32ms|624.86ms|109.01ms|356.67ms|332.29ms|
|censor 单op terror|all.tsv|10|100.00%|6.799933/s|2.62s|2.5s|2.72s|129.75ms|825.82ms|1.28s|
|censor 单op terror|terror-normal.tsv|1|100.00%|2.599984/s|622.66ms|590.56ms|685.44ms|212.67ms|341.94ms|374.26ms|
|censor 单op terror|terror-normal.tsv|10|100.00%|4.499954/s|2.57s|2.49s|2.72s|238.61ms|2.2s|1.96s|
|censor 单op terror|terror.tsv|1|100.00%|2.699977/s|650.61ms|394.65ms|765.9ms|204.08ms|301.31ms|345.84ms|
|censor 单op terror|terror.tsv|10|100.00%|4.699971/s|2.51s|2.36s|2.69s|363.11ms|2.1s|1.87s|
|censor 单op politician|all.tsv|1|100.00%|4.599953/s|770.04ms|388.27ms|931.15ms|48.93ms|146.55ms|216.48ms|
|censor 单op politician|all.tsv|10|100.00%|17.299827/s|1.43s|1.07s|2.48s|54.87ms|403.5ms|535.11ms|
|censor 单op politician|politician-normal.tsv|1|100.00%|0.99999/s|4.24s|2.38s|6.09s|107.81ms|202.87ms|979.25ms|
|censor 单op politician|politician-normal.tsv|10|100.00%|4.499957/s|4.77s|3.38s|6.75s|77.39ms|414.05ms|1.13s|
|censor 单op politician|politician.tsv|1|100.00%|4.599953/s|378.99ms|361.95ms|867.34ms|106.43ms|164.63ms|213.88ms|
|censor 单op politician|politician.tsv|10|100.00%|12.399926/s|1.02s|981.54ms|1.19s|171.85ms|779.64ms|776.69ms|
|eval politician.evalFacexDetect|all.tsv|1|100.00%|10.599896/s|183.18ms|150.33ms|199.01ms|38.09ms|79.05ms|93.35ms|
|eval politician.evalFacexDetect|all.tsv|10|100.00%|19.299853/s|604.5ms|592.76ms|672.81ms|48.41ms|514ms|506.42ms|
|eval pulp.evalPulp|all.tsv|1|100.00%|5.599954/s|404.7ms|342.61ms|450.62ms|79.82ms|122.46ms|177.92ms|
|eval pulp.evalPulp|all.tsv|10|100.00%|8.299905/s|1.36s|1.31s|1.42s|85.35ms|1.13s|1.1s|
|eval terror.evalTerrorClassify|all.tsv|1|100.00%|11.999878/s|173.01ms|141.58ms|203.88ms|25.91ms|75ms|83.09ms|
|eval terror.evalTerrorClassify|all.tsv|10|100.00%|24.19983/s|490.25ms|475.07ms|561.67ms|48.15ms|405.86ms|403.04ms|
|eval terror.evalTerrorDetect|all.tsv|1|100.00%|4.699957/s|292.03ms|286ms|317.22ms|130.14ms|205.55ms|211.54ms|
|eval terror.evalTerrorDetect|all.tsv|10|100.00%|5.599938/s|1.93s|1.9s|1.96s|204.95ms|1.75s|1.63s|
|eval terror.evalTerrorPostDetect|all.tsv|1|100.00%|9.599904/s|200.43ms|173.52ms|316.5ms|32.36ms|89.89ms|102.55ms|
|eval terror.evalTerrorPostDetect|all.tsv|10|100.00%|16.599845/s|691.7ms|657.27ms|770.42ms|61.65ms|591.84ms|582.4ms|
|eval terror.evalTerrorPredetect|all.tsv|1|100.00%|8.099937/s|224.72ms|175.17ms|259.26ms|64.71ms|113.83ms|123.19ms|
|eval terror.evalTerrorPredetect|all.tsv|10|100.00%|12.19988/s|949.06ms|920.78ms|1.05s|72.33ms|800.63ms|783.38ms|
