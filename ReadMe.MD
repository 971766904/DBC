# JTEXT上基于破裂危害等级的破裂预测
## file introduce
### 1.shot_select & signal_plot
    choose the shots with duration more than 50 ms and the disruption shot  
    should not be intentional disruption. Then it scans the valid signal and  
    valid shots. The next part is calculating std, mean and so on.

## 概念
1. 破裂危害和破裂物理不是重叠的，用破裂危害有限的跑，也可以覆盖很大范围破裂机理的区间
2. 用来指导训练集生成