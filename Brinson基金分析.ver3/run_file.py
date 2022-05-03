#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Brinson_fund_analysis import * 
import os

# 设置基本参数
BASIC_INDEX = "000300.XSHG"   # 采用沪深300作为基准指数

BEGIN = '2014-04-01'
END = '2022-01-01'
DATE = "2020-01-01"  # 因为对于申万一级行业指数曾经发生过改变，故需取最近的表格，这里选用DATE时的行业分类，后续可能会改为根据日期调整
for FUND_NAME in ['000011', '000592', '000689', '003567']:

    model = Brinson_analysis( BASIC_INDEX, FUND_NAME, BEGIN, END, DATE)
    # 进行brinson收益分解
    model.multi_periods_brison()  
    # 导出结果展示
    fig_perform_compar, fig_fund_perform, brinson_period, brinson_whole, fig_return_decompose, industries_panel, fig_diff = model.result_display()

    # 保存相关结果，不需要的可以注释掉，所以没有放在函数里了
    path = 'result/' + model.FUND_NAME
    if not os.path.exists(path):
        os.makedirs(path)
    file_name_prefix = path + '/' + model.FUND_NAME
    brinson_period.to_csv(file_name_prefix + '_历年brinson单期分析.csv',encoding="utf_8_sig")
    brinson_whole.to_csv(file_name_prefix + '_2014年至今总体brinson分析.csv',encoding="utf_8_sig")
    industries_panel.to_csv(file_name_prefix + '_基金基准的配置差.csv',encoding="utf_8_sig")
    fig_perform_compar.savefig(file_name_prefix + '_基金基准业绩比较.png')        
    fig_fund_perform.savefig(file_name_prefix + '_历年基金业绩.png')        
    fig_return_decompose.savefig(file_name_prefix + '_Brinson收益分解.png')       
    fig_diff.savefig(file_name_prefix + '_基金基准的配置差.png')


# In[ ]:




