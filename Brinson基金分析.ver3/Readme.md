brinson模型可以将超过基准收益的超额收益部分再分解为主动配置收益、标的选择收益、互动效应这三个部分。这里的归因方式为分类资产归因，以股票行业配置为归因对象。

总超额收益：策略相对于基准获得的额外收益，是下面主动配置收益、标的选择收益以及交互效应收益的汇总。

主动配置收益：主动配置的收益来源于对上涨行业的超配或对下跌行业的低配，是衡量对大类资产强弱走势进行判断的能力。如果大于零则意味着看准了市场大方向，并且高配了好的资产。

标的选择收益：标的选择的收益来源于对行业中表现好的个股的超配或对行业中表现差个股的低配。是对能否选出高于市场基准收益的资产，即在相同资金分配比例下，能否获得更高的收益能力的衡量。如果大于零则意味着拥有高于市场的个股选择能力。

互动收益：在总超额收益中，除去主动配置收益和标的选择收益，也就是超额收益中同时收到主动配置与标的选择影响的部分，就是互动收益。

下面来用公式具体的计算Brinson模型，但在计算这几部分收益之前，需要先构建这四个概念性组合：

在计算这几部分收益之前，需要先构建这四个概念性组合。

Q1 ：基准组合收益（基准收益*基准资产）

Q2 ：主动配置组合（基准收益*组合权重）

Q3 ：个股选择组合（组合收益*基准权重）

Q4 ：实际组合收益（组合收益*组合资产）

其中每个组合的计算可以用下面图表来表示
| Brinson分解表  |  组合资产i收益<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{130}\bg{black}r_{p,i}"  />  |  基准资产i收益<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{130}\bg{black}r_{b,i}"  />  |
|  ------  | ------- | ------- |
|  组合资产i权重<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{130}\bg{black}w_{p,i}" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}w_{p,i}" />  |  <img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}Q_{4}&space;=&space;\sum_{i=1}^{n}(w_{p,i}*r_{p,i})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}Q_{4} = \sum_{i=1}^{n}(w_{p,i}*r_{p,i})" />  |<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}Q_{2}&space;=&space;\sum_{i=1}^{n}(w_{p,i}*r_{b,i})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}Q_{2} = \sum_{i=1}^{n}(w_{p,i}*r_{b,i})" />
|  基准资产i权重<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{130}\bg{black}w_{b,i}" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}w_{b,i}" />  |  <img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}Q_{3}&space;=&space;\sum_{i=1}^{n}(w_{b,i}*r_{p,i})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}Q_{3} = \sum_{i=1}^{n}(w_{b,i}*r_{p,i})" />  | <img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{110}\bg{black}Q_{4}&space;=&space;\sum_{i=1}^{n}(w_{b,i}*r_{b,i})" title="https://latex.codecogs.com/png.image?\inline \large \dpi{110}\bg{black}Q_{4} = \sum_{i=1}^{n}(w_{b,i}*r_{b,i})" />   |


主动配置收益：<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{120}\bg{black}AR&space;=&space;Q_{2}-Q_{1}=\sum_{i=1}^{n}{w_{p,i}*r_{b,i}}&space;-&space;\sum_{i=1}^{n}{w_{b,i}*r_{b,i}}=&space;\sum_{i=1}^{n}{(w_{p,i}-w_{b,i})*r_{b,i}}" />

标的选择收益：<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{120}\bg{black}SR&space;=&space;Q_{3}-Q_{1}=\sum_{i=1}^{n}{w_{b,i}*r_{p,i}}&space;-&space;\sum_{i=1}^{n}{w_{b,i}*r_{b,i}}=&space;\sum_{i=1}^{n}{(r_{p,i}-r_{b,i})*w_{b,i}}" />

互动收益：<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{120}\bg{black}SR&space;=&space;Q_{4}-Q_{3}-Q_{2}&plus;Q_{1}=&space;\sum_{i=1}^{n}{(w_{p,i}-w_{b,i})*(r_{p,i}-r_{b,i})}"  />

总超额收益：<img src="https://latex.codecogs.com/png.image?\inline&space;\large&space;\dpi{120}\bg{black}TR&space;=&space;Q_{4}-Q_{1}&space;=&space;SR&plus;AR&plus;IR"  />

>**参考文献**：

>https://www.joinquant.com/view/community/detail/7b3bfc1d41e12d9d68213fab58937b95?type=1

Brinson基金分析模板

>https://www.joinquant.com/view/community/detail/da9fcadd00b27dcf92dca2a2999a0309?type=1

【归因分析讲解】之『Brinson模型介绍』