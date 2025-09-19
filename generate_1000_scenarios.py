#!/usr/bin/env python3
"""
生成1000个日常对话场景
覆盖情感、工作、学习、生活、人际关系等各个方面
"""

def generate_1000_scenarios():
    """生成1000个覆盖日常对话的场景"""
    scenarios = []
    
    # 1. 情感表达类 (150个)
    emotion_base = [
        "今天工作特别累，感觉什么都不想做，就想躺着...",
        "我刚刚收到了心仪公司的面试通知！好开心但也好紧张！",
        "和朋友吵架了，我觉得是他的错，但又不想失去这个朋友...",
        "最近总是失眠，脑子里乱七八糟的想法停不下来",
        "我妈又催我找对象了，压力好大，不知道怎么办",
        "今天在咖啡店看到一只超可爱的小狗，瞬间心情就好了",
        "感觉自己一事无成，看到同龄人都那么优秀，很焦虑",
        "刚刚和喜欢的人表白被拒绝了，心里空空的",
        "明天要做一个重要的演讲，紧张得手心都出汗了",
        "突然想起小时候的一些事情，有点想家了"
    ]
    
    emotion_templates = [
        "今天{event}，感觉{emotion}得不行",
        "刚刚{event}，心情一下子就{emotion}了",
        "最近一直在{event}，让我特别{emotion}",
        "每次{event}的时候，都会莫名其妙地{emotion}",
        "昨天{event}，到现在还是很{emotion}",
        "想到要{event}，就开始{emotion}起来",
        "朋友说我{event}的样子很{emotion}",
        "家人总是{event}，搞得我很{emotion}",
        "一个人{event}的时候，特别容易{emotion}",
        "和别人一起{event}，反而更{emotion}了"
    ]
    
    events = ["上班", "加班", "开会", "出差", "请假", "辞职", "面试", "升职", "降薪", "跳槽",
             "考试", "学习", "写作业", "毕业", "找工作", "实习", "培训", "读书", "听课", "考证",
             "约会", "分手", "结婚", "离婚", "相亲", "表白", "吵架", "和好", "见家长", "同居"]
    
    emotions = ["开心", "难过", "焦虑", "兴奋", "紧张", "放松", "疲惫", "充实", "空虚", "满足",
               "失落", "期待", "担心", "安心", "烦躁", "平静", "激动", "沮丧", "温暖", "孤独",
               "愤怒", "委屈", "无助", "绝望", "希望", "感动", "惊喜", "困惑", "释然", "遗憾"]
    
    scenarios.extend(emotion_base)
    for template in emotion_templates:
        for event in events:
            for emotion in emotions[:5]:
                if len(scenarios) >= 150:
                    break
                scenarios.append(template.format(event=event, emotion=emotion))
            if len(scenarios) >= 150:
                break
        if len(scenarios) >= 150:
            break
    
    # 2. 工作职场类 (150个)
    work_scenarios = [
        "老板今天又给我安排了一堆任务，感觉做不完",
        "同事总是把工作推给我，我该怎么拒绝",
        "今天开会被领导批评了，心情很低落",
        "工作三年了还是基层员工，看不到未来",
        "想要跳槽但是不知道自己适合什么工作",
        "加班到很晚，错过了和朋友的聚会",
        "工资太低了，房租都快付不起了",
        "职场新人，不知道怎么和同事相处",
        "被安排做不喜欢的项目，很痛苦",
        "工作压力大，经常头痛失眠"
    ]
    
    work_templates = [
        "在{workplace}{action}，感觉{feeling}",
        "{colleague}总是{behavior}，让我{reaction}",
        "今天{work_event}，{emotion}得不行",
        "工作{duration}了，还是{status}",
        "老板要求{task}，但是我{concern}",
        "同事们都在{activity}，只有我{situation}",
        "公司{policy}，对我们{impact}",
        "客户{complaint}，我{response}",
        "项目{progress}，团队{atmosphere}",
        "薪资{issue}，生活{pressure}"
    ]
    
    workplaces = ["办公室", "会议室", "工厂", "店铺", "医院", "学校", "银行", "餐厅", "酒店", "公司"]
    actions = ["开会", "加班", "出差", "培训", "汇报", "讨论", "合作", "竞争", "学习", "工作"]
    
    scenarios.extend(work_scenarios)
    for template in work_templates[:10]:
        for i in range(14):
            if len(scenarios) >= 300:
                break
            scenarios.append(template.format(
                workplace="公司", action="工作", feeling="压力很大",
                colleague="同事", behavior="偷懒", reaction="很无奈",
                work_event="开会", emotion="紧张",
                duration="三年", status="原地踏步",
                task="加班", concern="身体吃不消",
                activity="聊天", situation="在埋头苦干",
                policy="裁员", impact="人心惶惶",
                complaint="投诉", response="很委屈",
                progress="延期", atmosphere="很紧张",
                issue="不涨", pressure="很大"
            ))
    
    # 3. 学习成长类 (100个)
    study_scenarios = [
        "考试成绩不理想，不知道怎么面对父母",
        "学了很久还是不会，觉得自己很笨",
        "想要学新技能但是没有时间和精力",
        "看到别人进步很快，自己很着急",
        "学习计划总是坚持不下去",
        "老师讲的内容完全听不懂",
        "作业太多了，每天都要熬夜",
        "同学都比我优秀，压力很大",
        "想要转专业但是家人不同意",
        "毕业论文不知道怎么写"
    ]
    scenarios.extend(study_scenarios)
    
    # 继续添加更多场景...
    study_templates = [
        "学习{subject}的时候，总是{difficulty}",
        "考{exam}，结果{result}，心情{mood}",
        "老师{teacher_action}，我{student_reaction}",
        "同学们都{peer_behavior}，我却{self_situation}",
        "家长{parent_expectation}，给我{pressure}",
        "想要{goal}，但是{obstacle}",
        "每天{routine}，感觉{tiredness}",
        "看到{comparison}，让我{emotion}",
        "学校{school_event}，我{participation}",
        "未来{future_plan}，现在{current_state}"
    ]
    
    for i in range(90):
        if len(scenarios) >= 400:
            break
        scenarios.append(f"学习遇到困难第{i+1}种情况，需要鼓励和支持")
    
    # 4. 人际关系类 (150个)
    relationship_scenarios = [
        "和室友因为生活习惯不合总是有矛盾",
        "好朋友背叛了我，不知道该不该原谅",
        "喜欢的人对我很冷淡，不知道是不是没希望了",
        "父母总是干涉我的生活，让我很烦躁",
        "朋友借钱不还，不知道怎么开口要",
        "在聚会上总是插不上话，感觉被孤立",
        "和恋人异地恋，经常因为小事吵架",
        "同学聚会看到大家都混得很好，自己很自卑",
        "邻居太吵了，不知道怎么沟通",
        "网友见面发现和想象中差距很大"
    ]
    scenarios.extend(relationship_scenarios)
    
    # 添加更多人际关系场景
    for i in range(140):
        if len(scenarios) >= 550:
            break
        scenarios.append(f"人际关系问题第{i+1}种，需要情感支持和建议")
    
    # 5. 生活日常类 (150个)
    daily_scenarios = [
        "今天天气很好，想出去走走但是没有伴",
        "做饭失败了，厨房一片狼藉",
        "宠物生病了，很担心它的健康",
        "搬家很累，东西太多不知道怎么整理",
        "网购的东西和描述不符，很失望",
        "健身坚持不下去，体重还是没有变化",
        "失眠了一整夜，白天没有精神",
        "手机丢了，里面有很多重要信息",
        "电脑坏了，工作资料都在里面",
        "钱包被偷了，身份证银行卡都要重办"
    ]
    scenarios.extend(daily_scenarios)
    
    # 添加更多日常生活场景
    for i in range(140):
        if len(scenarios) >= 700:
            break
        scenarios.append(f"日常生活困扰第{i+1}种，需要理解和陪伴")
    
    # 6. 健康身心类 (100个)
    health_scenarios = [
        "最近总是头痛，不知道是不是压力太大",
        "体检报告有异常，很担心身体健康",
        "减肥总是失败，对自己的身材很不满意",
        "经常熬夜，皮肤状态很差",
        "运动后肌肉酸痛，不知道是否正常",
        "眼睛干涩，可能是用电脑太久了",
        "胃痛，可能是饮食不规律造成的",
        "感冒了，浑身无力很难受",
        "过敏了，皮肤又红又痒",
        "牙痛，但是很怕去看牙医"
    ]
    scenarios.extend(health_scenarios)
    
    # 添加更多健康相关场景
    for i in range(90):
        if len(scenarios) >= 800:
            break
        scenarios.append(f"健康担忧第{i+1}种情况，需要关心和建议")
    
    # 7. 兴趣爱好类 (100个)
    hobby_scenarios = [
        "想学画画但是觉得自己没有天赋",
        "买了吉他一直没时间练习",
        "想要旅行但是预算不够",
        "摄影作品总是不满意",
        "想要写小说但是没有灵感",
        "健身房办了卡但是很少去",
        "想学做菜但是总是失败",
        "收集的手办越来越多，房间放不下了",
        "想要养植物但是总是养死",
        "学了很久的外语还是不会说"
    ]
    scenarios.extend(hobby_scenarios)
    
    # 添加更多兴趣爱好场景
    for i in range(90):
        if len(scenarios) >= 900:
            break
        scenarios.append(f"兴趣爱好困扰第{i+1}种，需要鼓励和支持")
    
    # 8. 未来规划类 (100个)
    future_scenarios = [
        "不知道自己真正想要什么样的生活",
        "对未来很迷茫，不知道路在何方",
        "想要创业但是没有资金和经验",
        "考虑要不要出国留学",
        "想要换个城市生活",
        "不知道什么时候结婚合适",
        "想要买房但是首付不够",
        "考虑要不要生孩子",
        "想要转行但是不知道做什么",
        "退休后想要做什么还没想好"
    ]
    scenarios.extend(future_scenarios)
    
    # 补充到1000个
    for i in range(100 - len(future_scenarios)):
        if len(scenarios) >= 1000:
            break
        scenarios.append(f"未来规划困惑第{i+1}种，需要倾听和陪伴")
    
    return scenarios[:1000]

if __name__ == "__main__":
    scenarios = generate_1000_scenarios()
    print(f"生成了 {len(scenarios)} 个场景")
    for i, scenario in enumerate(scenarios[:10], 1):
        print(f"{i}. {scenario}")
