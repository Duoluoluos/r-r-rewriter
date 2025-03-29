#SRT
SRT_instruction = """你学识渊博，请阅读这道保荐代表人考试试题，以及答案与解析，先一步一步地结合解析思考一下正确答案是如何得到的，然后再分析这道试题的考点。\n输出格式参考：\n解题过程复盘：\n... 考点解析：\n1、考点1；\n2、考点2；\n3、考点3\n """
SRT_few_shot_input = """下列交易或事项会计处理的表述中，正确的是（ ）。\nA对于附回售条款的股权投资，投资方除拥有与普通股股东一致的投票权及分红权等权利之外，还拥有一项回售权（例如投资方与被投资方约定，若被投资方未能满足特定目标，投资方有权要求按投资成本加年化8%收益的对价将该股权回售给被投资方，8%假设代表被投资方在市场上的借款利率水平），投资方对被投资方没有重大影响。被投资方将附回售条款的股权投资分类为权益工具\nB考虑公允价值变动对净利润的影响，爱德华公司将持有普朗克公司 8%的股权投资从以公允价值计量且其变动计入当期损益的金融资产，重分类为以公允价值计量且其变动计入其他综合收益的金融资产\nC企业持有的将来须用或可用企业自身权益工具进行结算的非衍生工具合同，且企业根据该合同将收到可变数量的自身权益工具，属于金融资产\nD因普朗克公司的信用等级下降，爱德华公司将持有并分类为以摊余成本计量的普朗克公司债券全部出售，同时将该类别的债权投资全部重分类为以公允价值计量且其变动计入其他综合收益的金融资产\nE企业将一项其他权益工具投资重分类为交易性金融资产的，应当继续以公允价值计量该金融资产\n \
正确答案：C\n
解析：选项A，从被投资方角度看，由于被投资方存在无法避免的向投资方交付现金的合同义务，应分类为金融负债进行会计处理。\n选项B，企业对相关金融资产进行重分类时，应考虑其业务模式是否改变，而不是出于对净利润的影响。\n选项C，企业持有的将来须用或可用企业自身权益工具进行结算的非衍生工具合同，且企业根据该合同将收到可变数量的自身权益工具，属于金融资产。\n选项D，企业改变其管理金融资产的业务模式时，应当按照规定对所有受影响的相关金融资产进行重分类，爱德华公司相关债券信用等级下降可能发生减值，并不意味着其改变金融资产的业务模式，不应将其进行重分类。\n选项E，对于非交易性权益工具投资不得进行重分类。
"""
SRT_few_shot_output = """解题过程复盘：
1. 选项A：题目描述的情境中，投资方虽然拥有回售权，但回售条款意味着被投资方有一项无法避免的义务，即在特定条件下按约定价格回购股权。这种情况下，从被投资方的角度来看，这项回购义务应被分类为金融负债。因此，选项A的表述错误。
2. 选项B：在企业对金融资产进行重分类时，应当基于企业的业务模式变化，而不是基于对净利润的影响来进行判断。也就是说，企业重分类金融资产时，必须有明确的业务模式改变作为依据，而不能仅仅因为考虑公允价值变动对净利润的影响进行重分类。因此，选项B的表述错误。
3. 选项C：根据题目中的描述，企业持有的合同如果是将来须用或可用企业自身权益工具进行结算，且合同要求企业接收可变数量的权益工具，这种情况确实应分类为金融资产。因此，选项C的表述是正确的。
4. 选项D：企业改变金融资产的业务模式时，确实应按照规定对所有受影响的金融资产进行重分类。然而，题目中提到的情况是由于信用等级下降可能导致资产减值，而这并不构成改变业务模式的依据，因此，不应因此将金融资产重分类。因此，选项D的表述错误。
5. 选项E：根据会计准则，企业持有的非交易性权益工具投资不允许进行重分类。因此，选项E的表述错误。
通过以上分析，正确答案是 C。
考点解析：
1. 考点1：金融负债和权益工具的分类。本题涉及附回售条款的股权投资，这种情况下应根据是否存在无法避免的合同义务来判断是否应分类为金融负债。理解金融负债和权益工具的分类原则是解答此类问题的关键。
2. 考点2：金融资产重分类的条件。在金融资产的重分类中，必须明确业务模式的改变才可以进行重分类。考生需掌握金融资产分类的标准和重分类的条件，这是本题考察的核心。
3. 考点3：金融工具的会计处理。题目考察了企业持有的合同是否应分类为金融资产的判断标准，特别是与自身权益工具结算相关的合同处理，理解这些会计处理的原则对于解答此类题目非常重要。
"""

Syllabus_instruction = """You are knowledgeable, please read the following question and its answer, and think step by step how the correct answer is obtained, and then analyze the key points of this question. The output format is as follows:\nStep-by-step thinking: ...\nKey point analysis:\n1. Key point 1;\n2. Key point 2;\n3. Key point 3"""
Syllabus_few_shot_input = """Question: Will there be mental health services if student need them?\nAnswer: Schools will have staff to help students with mental health problems."""
Syllabus_few_shot_output = """Step-by-step thinking:
1. Understanding the Question: The question asks whether mental health services will be available for students if they need them. This implies a concern for the accessibility and support of mental health resources within a school setting.
2. Identifying the Response: The answer states that schools will have staff to assist students with mental health issues. This suggests that there are provisions in place for addressing such needs.
3. Inferring the Implications: The mention of "staff to help" implies the existence of dedicated resources, such as counselors, psychologists, or trained teachers who can provide support, indicating that mental health services are indeed available.
4. Concluding the Answer: By confirming that there is staff available, the answer reassures that mental health services will be accessible to students when required.
Key point analysis:
1. Availability of Services: The question highlights the importance of having mental health services accessible to students, indicating a need for schools to prioritize mental health.
2. Role of School Staff: The answer emphasizes the role of school staff in providing mental health support, indicating that trained personnel are essential in addressing student needs.
3. Proactive Support System: The presence of mental health staff suggests a proactive approach to student wellbeing, which is crucial for early intervention and ongoing support for mental health issues.
"""

Fintext_instruction = """You are knowledgeable, please read the following question and its answer, and think step by step how the correct answer is obtained, and then analyze the key points of this question. The output format is as follows:\nStep-by-step thinking: ...\nKey point analysis:\n1. Key point 1;\n2. Key point 2;\n3. Key point 3"""
Fintext_few_shot_input = """Question: Are transactions signed between 12 July 2023 and 12 October 2023 (and implemented on 12 October 2023 or later) subject to mandatory notification under Regulation EU 2022/2560?\nAnswer: Notifiable concentrations under Article 20 of Regulation EU 2022/2560 for which the agreement was concluded on 12 July 2023 or later but which have not yet been implemented on 12 October 2023, will need to be notified pursuant to Article 21 of Regulation EU 2022/2560 and are subject to the standstill obligation under Article 24 of Regulation EU 2022/2560. By contrast, the notification obligation does not apply to concentrations for which the agreement was concluded on 12 July 2023 or later but which are implemented before 12 October 2023. Notifying Parties are encouraged to engage in pre-notification contacts, in principle as of September 2023, in advance to facilitate the submission of notifications as from 12 October 2023."""
Fintext_few_shot_output = """Step-by-step thinking:
1. Understanding the Question: The question asks about the status of transactions between parties conducted after 12 July 2023 and before 12 October 2023, specifically those involving notifiable concentrations. It also mentions that these transactions were agreed upon on 12 July 2023 or earlier.
2. Identifying the Response: The answer states that notifiable concentrations will need to be notified pursuant to Article 21 of Regulation EU 2022/2560 and are subject to the standstill
obligation under Article 24 of Regulation EU 2022/2560. However, it should be noted that the notification obligation does not apply to concentrations for which
the agreement was concluded on 12 July 2023 or later but which are implemented before 12 October 2023.
3. Inferring the Implications: The response indicates that the parties involved in the transaction may benefit from being able to notify their respective counterparties ahead of time through pre-notification contacts.
4. Concluding the Answer: Based on the information provided, the answer provides guidance regarding the necessary steps needed to comply with the regulations related to notifiable concentrations.
Key point analysis:
1. Mandatory Notification: The answer clarifies that notifiable concentrations must be notified pursuant to Article 21 of Regulation EU 2022/2560.
2. Standstill Obligation: The answer mentions the standstill obligation under Article 24 of Regulation EU 2022/2560, which requires the parties to hold off any further transactions until the notification has been received.
3. Pre-Notification Contacts: The answer suggests that parties can engage in pre-notification contacts, in principle
as of September 2023, to facilitate the submission of notifications as from 12 October 2023.
"""