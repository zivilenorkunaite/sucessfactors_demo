"""
Application Configuration
Centralized configuration for catalog and schema names.

Update these values to change the default catalog and schema used across all notebooks.
"""

# Unity Catalog Configuration
DEFAULT_CATALOG_NAME = "demos"
DEFAULT_SCHEMA_NAME = "career_path_temp"


# MLflow Experiment Configuration
MLFLOW_EXPERIMENT_PATH = "/Shared/career_path_intelligence_engine/career_intelligence_models_updated"

# Data Product Table Names (SAP SuccessFactors Data Products via Delta Sharing)
EMPLOYEES_DATA_PRODUCT_TABLE = "core_workforce_data_dp.coreworkforcedata.coreworkforce_standardfields"
PERFORMANCE_DATA_PRODUCT_TABLE = "performance_data_dp.performancedata.performancedata"
LEARNING_DATA_PRODUCT_TABLE = "learning_history_dp.learninghistory.learningcompletion"

# Department Code to Name Mapping
# Maps SAP SuccessFactors department codes to human-readable department names
# Based on job title analysis from actual data
DEPARTMENT_CODE_TO_NAME = {
    # Original departments
    1034: 'Engineering',
    1035: 'Product',
    1036: 'Sales',
    1037: 'Marketing',
    1038: 'Finance',
    1039: 'HR',
    1040: 'Operations',
    1041: 'Legal',
    
    # Development departments (Development Analyst, Development Manager, Development Analyst Lead)
    18: 'Development',
    812556: 'Development',
    812579: 'Development',
    812602: 'Development',
    
    # Manufacturing departments (Assembly Worker, Assembly Manager, Capacity Planning Manager)
    812535: 'Manufacturing',
    812558: 'Manufacturing',
    812581: 'Manufacturing',
    812650: 'Manufacturing',
    
    # Production departments (Production Director, Production Technician)
    812604: 'Production',
    
    # HR/Recruiting departments
    812617: 'HR',  # Recruiting Manager, Program Manager, VP Human Resources USA, Executive Assistant
    812553: 'Recruiting',  # Recruiter, Recruiting Manager, Sr Recruiter
    812576: 'Recruiting',  # Recruiter, Recruiting Manager, Sr Recruiter
    812599: 'Recruiting',  # Recruiter
    812550: 'HR',  # HR Business Partner, HR Business Office Director UK
    812554: 'Compensation',  # Compensation Manager, Sr. Compensation Analyst
    812577: 'Compensation',  # Compensation Manager, Sr. Compensation Analyst
    812571: 'HR',  # VP Human Resources FR
    812573: 'HR',  # HR Business Partner, Payroll Admin, HR Business Office Director FR, HR Project Manager, VP Human Resources FR, Executive Assistant
    812548: 'HR',  # VP Human Resources UK, Payroll Admin, Executive Assistant
    812552: 'Talent Management',  # Director of Talent Management
    812575: 'Talent Management',  # Director of Talent Management
    812598: 'Talent Management',  # Director of Talent Management
    812549: 'Learning & Development',  # Learning Business Partner
    812551: 'HR',  # External Workforce Director
    
    # Quality Assurance departments (Inspector, Quality Assurance Manager, QA Engineer, QA Administrator)
    812629: 'Quality Assurance',
    812511: 'Quality Assurance',
    812557: 'Quality Assurance',  # Quality Assurance Manager, QA Engineer, QA Administrator, Supply Chain Director, Craft Workers, Professional, Executive Assistant
    812583: 'Quality Assurance',  # QA Engineer
    
    # Operations departments (SVP Operations & Maintenance, VP Operations, Project Execution Lead, Project Manager, Program Management Office, Scheduler, Planner, Planning & Scheduling Manager, Administrator)
    30: 'Operations',
    812580: 'Operations',  # SVP Operations & Maintenance, VP Operations, Project Execution Lead, Project Manager, Program Management Office, Scheduler, Quality Assurance Manager, Administrator
    812603: 'Operations',  # VP Operations, Project Execution Lead, Program Management Office, Executive Assistant
    1: 'Operations',  # VP Operations
    
    # Engineering departments (Engineer I/II/III, Engineering Manager, Engineering Intern)
    812562: 'Engineering',
    812585: 'Engineering',
    812608: 'Engineering',
    
    # Planning & Scheduling departments
    812561: 'Planning & Scheduling',  # Planning & Scheduling Manager, Scheduler, Planner
    812584: 'Planning & Scheduling',  # Planning & Scheduling Manager
    
    # Facilities departments (VP Facilities, Facilities Manager, Building Manager, Custodian)
    812559: 'Facilities',  # VP Facilities
    812582: 'Facilities',  # Facilities Manager
    812605: 'Facilities',  # Facilities Manager
    812565: 'Facilities',  # Building Manager, Custodian
    812588: 'Facilities',  # Custodian
    812611: 'Facilities',  # Custodian, Building Manager
    
    # Sales departments
    812591: 'Sales',  # SVP Sales
    
    # IT departments
    812592: 'Information Technology',  # VP Information Technology
    
    # Executive departments (President, Executive Assistant to the President)
    812627: 'Executive',  # Executive Assistant, Assembly Worker
    812570: 'Executive',  # President France, Executive Assistant
    21: 'Executive',  # President United States, Executive Assistant to the President, Compensation Manager
    812662: 'Executive',  # President BestRun EMEA
    812547: 'Executive',  # President United Kingdom
    812593: 'Executive',  # President Germany, Executive Assistant to the President
    
    # Project Management departments
    7181698: 'Project Management',  # Project Manager
    
    # Additional Development departments
    812510: 'Development',  # Development Analyst, Development Manager, Program Manager
    812533: 'Development',  # Development Analyst, Development Analyst Lead, Development Manager
    812648: 'Development',  # Development Manager, Program Manager, Development Analyst Lead
    
    # Additional Engineering departments
    9: 'Engineering',  # Engineering
    812516: 'Engineering',  # Engineer I, QA Administrator
    812539: 'Engineering',  # Electrical Engineer III, Engineering Intern, QA Engineer
    812631: 'Engineering',  # Engineering Manager, Engineering Intern, Engineer III
    812654: 'Engineering',  # Engineering Intern, Engineer II, Engineer III
    1280288: 'Engineering',  # Engineering Intern, Engineer III, Digital Expert
    3149666: 'Engineering',  # Engineer II, Engineer I
    5190066: 'Engineering',  # Engineer Intern, Engineer I, Engineer II
    7549633: 'Engineering',  # Engineering Manager
    7999619: 'Engineering',  # Engineering Manager
    1446880708034: 'Engineering',  # Engineering Manager, Engineer III, Engineer I
    
    # Additional HR departments
    812502: 'HR',  # VP Human Resources
    812504: 'HR',  # External Workforce Director, Executive Assistant, HR Business Office Director CN
    812507: 'HR',  # Recruiting Manager, Recruiter, Sr Recruiter
    812527: 'HR',  # HR Business Office Manager, HR Business Partner, Payroll Admin
    812529: 'HR',  # Managing Director Human Resources
    812530: 'HR',  # Recruiter, Recruiting Manager, Sr Recruiter
    812544: 'HR',  # Managing Director Finance, Managing Director HR Systems, HR systems specialist
    812594: 'HR',  # VP Human Resources Germany
    812596: 'HR',  # HR Business Office Director DE, HR Project Manager, HR Business Partner
    812600: 'HR',  # Compensation Manager, Sr. Compensation Analyst
    812640: 'HR',  # VP Human Resources Brazil
    812642: 'HR',  # Payroll Admin, HR Business Partner, IT Business Partner
    812645: 'HR',  # Recruiter, Recruiting Manager, Sr Recruiter
    1233984: 'HR',  # Payroll Admin, Compensation Manager, Learning BP Check
    3269596: 'HR',  # 人事ビジネスパートナ
    6329900: 'HR',  # HR Business Partner Mexico
    6721599: 'HR',  # شريك أعمال الموارد البشرية السعودية
    6976612: 'HR',  # الموارد البشرية الإمارات العربية المتحدة شريك الأعمال في
    7181691: 'HR',  # HR Business Partner
    7549630: 'HR',  # VP Human Resources NL
    7549637: 'HR',  # Payroll Admin, HR Business Partner, HR Business Office Director
    7570210: 'HR',  # HR Business Partner Italy
    7720083: 'HR',  # HR Business Partner Argentina
    7999611: 'HR',  # HR Business Partner, HR Business Office Director, Payroll Admin
    7999617: 'HR',  # VP Human Resources PT
    8070652: 'HR',  # HR Business Partner Chile
    8119620: 'HR',  # HR Business Partner Colombia
    8131485: 'HR',  # HR Business Partner CHE, Payroll Manager
    8849635: 'HR',  # HR Business Partner CZE
    10271151: 'HR',  # HR Business Partner Spain
    11049776: 'HR',  # HR BP Maintenance, HR Manager, HR BP Market Research
    1446868379564: 'HR',  # HR Business Partner Ireland
    1446869919021: 'HR',  # HR Director (DE), HRBP Sales & Services, HR Professional
    1446870018524: 'HR',  # HR Business Office Director (BE), HR Business Partner (BE), Payroll Admin (BE)
    1446870018528: 'HR',  # VP Human Resources (BE)
    1446870098531: 'HR',  # 인재관리 (한국), 인사행정 (한국), 채용담당 (한국)
    1446880708037: 'HR',  # Payroll Admin, VP Human Resources SE, HR Business Office Director
    6721585: 'HR',  # شريك أعمال الموارد البشرية قطر
    
    # Additional Learning & Development departments
    11: 'Learning & Development',  # Director Center of Excellence, Learning Director, Director Learning
    812503: 'Learning & Development',  # Learning Business Partner
    812526: 'Learning & Development',  # Learning Business Partner
    812595: 'Learning & Development',  # Head of Digital Curriculum
    812601: 'Learning & Development',  # Learning Business Partner
    812641: 'Learning & Development',  # Learning Business Partner
    7510281: 'Learning & Development',  # HR Business Partner Korea
    
    # Additional Talent Management departments
    15: 'Talent Management',  # Director of Talent Management, Chief Talent Officer, VP Talent Acquisition
    812506: 'Talent Management',  # Director of Talent Management
    812525: 'Talent Management',  # Director of Talent Management, Executive Assistant
    812644: 'Talent Management',  # Director of Talent Management
    5229562: 'Talent Management',  # Director of Talent Management
    5969572: 'Talent Management',  # Director of Talent Management
    
    # Additional Compensation/Total Rewards departments
    14: 'Compensation',  # Compensation Manager, TR Consultant II, Total Rewards Consultant II
    812508: 'Compensation',  # Compensation Manager, Sr. Compensation Analyst
    812531: 'Compensation',  # Compensation Manager, Sr. Compensation Analyst, Compensation Analyst
    812646: 'Compensation',  # Compensation Analyst, Compensation Manager
    
    # Additional Legal/Employee Relations departments
    12: 'Legal',  # European Labor Law, ER Consultant
    
    # Additional Organizational Development departments
    13: 'Organizational Development',  # Organizational Development Manager, OD Consultant
    
    # Additional Planning & Scheduling departments
    10: 'Planning & Scheduling',  # 计划和排班经理, Planner, Scheduler
    812538: 'Planning & Scheduling',  # Scheduler, Planning & Scheduling Manager, Senior Planner
    812630: 'Planning & Scheduling',  # Planning, Planner, Scheduler
    812653: 'Planning & Scheduling',  # Planning & Scheduling Manager, Scheduler
    3180005: 'Planning & Scheduling',  # Planning and Scheduling Manager, 東北支社長
    5190067: 'Planning & Scheduling',  # Planning and Scheduling Manager, Planner, Scheduler
    7549641: 'Planning & Scheduling',  # Planning & Scheduling Manager
    7999616: 'Planning & Scheduling',  # Planning & Scheduling Manager
    11049773: 'Planning & Scheduling',  # Planning Engineer, Scheduler, Craft Planner
    1446880708035: 'Planning & Scheduling',  # Planning & Scheduling Manager, Scheduler, Planner
    
    # Additional Operations departments
    8: 'Operations',  # VP Maintenance
    17: 'Operations',  # Management & Planning
    812534: 'Operations',  # Managing Director Sales, Sr. Director Maintenance, Sr. Director Operations
    812540: 'Operations',  # Management and Planning
    812626: 'Operations',  # Executive Assistant, Administrator, SVP Operations & Maintenance
    812649: 'Operations',  # Program Management Office, Administrator, Project Execution Lead
    1280285: 'Operations',  # VP Operations, Executive Assistant, SVP Operations & Maintenance
    1289589: 'Operations',  # VP Maintenance, Management & Planning, Executive Assistant
    1289592: 'Operations',  # SVP Operations & Maintenance, Executive Assistant, Management & Planning
    1289595: 'Operations',  # Executive Assistant, Quality Assurance Manager, Supply Chain Director
    1289598: 'Operations',  # Quality Assurance Manager, Executive Assistant, VP Operations
    1289601: 'Operations',  # VP Maintenance, Facilities Manager, SVP Procurement
    1389587: 'Operations',  # VP Operations, SVP Operations & Maintenance
    3149678: 'Operations',  # Production Oversight Manager, Administrator, Assembly Manager
    5190062: 'Operations',  # Assembly Manager, Production Director, Program Management Office
    5190063: 'Operations',  # VP Procurement, SVP Operations & Maintenance, VP Maintenance
    5969566: 'Operations',  # Project Execution Lead, Assembly Worker, Production Director
    5969569: 'Operations',  # VP Operations, SVP Operations & Maintenance
    6329894: 'Operations',  # Program Management Office, Production Director
    6329896: 'Operations',  # SVP Operations & Maintenance, VP Operations
    6780595: 'Operations',  # SVP Operations & Maintenance, VP Operations
    6780596: 'Operations',  # Program Management Office, Production Director
    7039583: 'Operations',  # VP Operations, SVP Operations & Maintenance
    7039584: 'Operations',  # Program Management Office, Production Director
    7181690: 'Operations',  # VP Operations, SVP Operations & Maintenance, Program Management Office
    7510282: 'Operations',  # VP Operations Korea, SVP Operations & Maintenance Korea
    7549631: 'Operations',  # VP Operations, SVP Operations & Maintenance, Program Management Office
    7570213: 'Operations',  # SVP Operations & Maintenance, VP Operations
    7720082: 'Operations',  # Production Director, Program Management Office
    7720085: 'Operations',  # SVP Operations & Maintenance, VP Operations
    7999622: 'Operations',  # SVP Operations & Maintenance, VP Operations, Program Management Office
    7999623: 'Operations',  # Production Director
    8070646: 'Operations',  # Production Director, VP Operations, Program Management Office
    8119617: 'Operations',  # SVP Operations and Maintenance, President Colombia
    8119623: 'Operations',  # VP Operations
    8119626: 'Operations',  # Director of Projects, Production Director
    8131479: 'Operations',  # VP Operations
    8131482: 'Operations',  # Program Management Office, Production Director
    8849629: 'Operations',  # SVP Operations and Maintenance CZE, President Czech Republic
    8849632: 'Operations',  # VP Operations CZE
    10271148: 'Operations',  # VP Operations, Production Director, Program & Management Office
    10810275: 'Operations',  # SVP Operations & Maintenance
    10810284: 'Operations',  # SVP Operations & Maintenance
    10810287: 'Operations',  # Digital Expert, Administrator, Assembly Manager
    10810293: 'Operations',  # Management & Planning
    10810299: 'Operations',  # Production Director
    1446868379562: 'Operations',  # Operations Manager, Google HR Professional, Program Management Office
    1446868379565: 'Operations',  # VP Operations, SVP Operations & Maintenance
    1446880708033: 'Operations',  # Program Management Office, Assembly Worker, Capacity Planning Manager
    1446880708038: 'Operations',  # VP Operations
    
    # Additional Manufacturing/Production departments
    812512: 'Manufacturing',  # 管理员, Capacity Planning Manager, Production Planning EngineerEngineering
    4829568: 'Manufacturing',  # Production Manager, SVP Hong Kong Operations & Maintenance, Payroll Manager
    6721598: 'Manufacturing',  # مهندس تخطيط الإنتاج, خبير تدريب, مدير التجميع
    6976610: 'Manufacturing',  # مدير الإنتاج, خبير التكنولوجيا, خبير تدريب
    11049767: 'Manufacturing',  # Production Oversight Manager, Assembly Worker, Production Technician
    1446869919026: 'Manufacturing',  # Head of Manufacturing, Production Oversight Manager, Professional
    
    # Additional Production departments
    4829607: 'Production',  # Supply Chain Director, HR Business Partner, Production Manager
    7549627: 'Production',  # Production Director
    7570211: 'Production',  # Production Director, Program Management Office
    10810260: 'Production',  # Program Management Office CZE, Production Director CZE
    1446870018521: 'Production',  # Production Director (BE), Engineer II (BE), VP Operations (BE)
    
    # Additional Quality Assurance departments
    812606: 'Quality Assurance',  # Quality Assurance Manager, QA Administrator
    812652: 'Quality Assurance',  # QA Administrator, Quality Assurance Manager, QA Engineer
    1280300: 'Quality Assurance',  # QA Manager
    5190064: 'Quality Assurance',  # Quality Assurance Manager
    4829574: 'Quality Assurance',  # Quality Assurance Manager
    
    # Additional Facilities departments
    812513: 'Facilities',  # Facilities Manager, Custodian, Building Manager
    812528: 'Facilities',  # (if exists)
    812536: 'Facilities',  # Facilities Manager
    812542: 'Facilities',  # Custodian, Building Manager
    812628: 'Facilities',  # Facilities Manager, Custodian, Security
    812634: 'Facilities',  # Waitstaff, Building Manager
    812657: 'Facilities',  # Custodian, Building Manager
    1280303: 'Facilities',  # VP Facilities, Custodian, VP Maintenance
    4829577: 'Facilities',  # Facilities Manager
    4829604: 'Facilities',  # VP Maintenance
    5190065: 'Facilities',  # Facilities Manager, Custodian
    5190068: 'Facilities',  # Building Manager
    
    # Additional Sales departments
    812614: 'Sales',  # SVP Sales
    812637: 'Sales',  # Sales Representative, Sales Manager-EAST, Account Executive
    1280297: 'Sales',  # SVP Sales
    11049785: 'Sales',  # Presales, VP Sales, Sales Director
    1446870398519: 'Sales',  # Sales Representative, Sales Manager
    1446870398523: 'Sales',  # Sales Rep 4, Sales Representative, Sales Manager
    1446890112523: 'Sales',  # Sales Associate, Retail Sales Associate, Retail Store Manager
    
    # Additional Information Technology departments
    812615: 'Information Technology',  # VP Information Technology
    812523: 'Information Technology',  # VP Information Technology
    812638: 'Information Technology',  # IT Manager, HR IT Specialist, Network Administrator
    1299967: 'Information Technology',  # IT Business Partner, VP Information Technology
    11049779: 'Information Technology',  # System Admin, CTO, HR Admin
    1446869919022: 'Information Technology',  # System Administrator, Executive Management, IT Project Manager
    
    # Additional Finance departments
    812636: 'Finance',  # Treasury Specialist, Treasury Subsidiary Manager, AR Accountant
    11049782: 'Finance',  # Head of Finance, Executive Assistant, Financial Controller
    1446869919019: 'Finance',  # Treasury Subsidiary Manager, Financial Controller, Purchasing Manager (DE)
    1446870368521: 'Finance',  # Finance Officer (also Operations)
    1446870368524: 'Finance',  # Finance Officer, Resource Manager, Director Operations (US)
    
    # Additional Supply Chain/Procurement departments
    4829571: 'Supply Chain',  # VP Procurement
    4829610: 'Supply Chain',  # 釆購副總裁
    6976609: 'Supply Chain',  # مدير سلسلة الإمداد, مدير التخطيط والجدولة, نائب الرئيس للعمليات
    6721587: 'Supply Chain',  # مدير الخدمات, مدير سلسلة الإمداد, نائب الرئيس للعمليات
    6721600: 'Supply Chain',  # مدير الخدمات, مدير سلسلة الإمداد, نائب الرئيس للعمليات
    1446876721665: 'Supply Chain',  # VP Warehousing, Inventory Controller, Warehouse Supervisor
    1446876721666: 'Supply Chain',  # VP Logistics
    1446876721667: 'Supply Chain',  # Purchaser
    
    # Additional Maintenance departments
    11049770: 'Maintenance',  # Maintenance Engineer II, Quality Controller, Inspector
    
    # Additional Executive departments
    20: 'Executive',  # CEO BestRun Corporation
    23: 'Executive',  # President China, Executive Assistant, Vice President Finance
    25: 'Executive',  # President Chile, President BestRun Americas
    26: 'Executive',  # President BestRun Asia Pacific, President Australia, 代表取締役社長
    27: 'Executive',  # Chief Operating Officer, Chief Information Officer (CIO), Executive Assistant
    28: 'Executive',  # Integration Administrator, Executive Assistant, Decentralised Administrator
    812524: 'Executive',  # Executive Assistant
    812639: 'Executive',  # President Brazil, Executive Assistant
    812663: 'Executive',  # Executive Assistant
    1233981: 'Executive',  # President Russia, Vice President Finance, Executive Assistant to the President
    5229565: 'Executive',  # Executive Assistant to the President, President Best Run South Africa
    5969563: 'Executive',  # President Best Run Canada, Executive Assistant to the President
    6329892: 'Executive',  # President Best Run Mexico
    6780598: 'Executive',  # President BestRun Poland, Payroll Manager
    7039582: 'Executive',  # President Best Run New Zealand
    7181695: 'Executive',  # President of BestRun Holdings, Executive Assistant to the President
    7510279: 'Executive',  # President BestRun Korea
    7570212: 'Executive',  # President BestRun Italy
    7720084: 'Executive',  # President BestRun Argentina
    7999621: 'Executive',  # President Portugal
    8131476: 'Executive',  # President Switzerland, SVP Operations and Maintenance CHE
    11509838: 'Executive',  # VP Corporate, CEO
    1446868379558: 'Executive',  # President Best Run Ireland
    1446869919028: 'Executive',  # Executive Management, VP Corporate, VP Products
    1446870098532: 'Executive',  # 베스트런한국 사장
    1446880708036: 'Executive',  # President SE
    3149675: 'Executive',  # President Singapore
    6721588: 'Executive',  # رئيس قطر, مدير الرواتب
    6721597: 'Executive',  # رئيس المملكة العربية السعودية, مدير الرواتب
    6976611: 'Executive',  # رئيس الامارات المتحدة العربية, مدير الرواتب
    
    # Additional Retail departments
    5749586: 'Retail',  # Planning, Retail SVP Sales and Marketing, Retail Forecasting
    5749589: 'Retail',  # Executive management, Retail HR Manager
    5749595: 'Retail',  # Promotion Associate
    5749601: 'Retail',  # Planning, Retail Category Manager, Retail Marketing Manager
    1446880696075: 'Retail',  # Retail Manager, Generation Manager, US Director-Utilities
    1446880696077: 'Retail',  # Marketing Representative-Customer Svcs, Billing Agent-Customer Svcs, Customer Service Representative-Customer Svcs
    1446880696078: 'Retail',  # T&D Field Manager, DE Director-Utilities, Customer Services Manager
    1446880696080: 'Retail',  # Billing Agent -Retail, Marketing Representative-Retail, Customer Service Representative-Retail
    
    # Additional Regional/Country-specific Sales departments (Japan)
    10079877: 'Sales',  # 西日本営業部長, アライアンス本部長, 南関東支社長
    10149877: 'Sales',  # マーケティング本部長
    10149880: 'Sales',  # ビジネスディベロップメント本部長
    10149883: 'Sales',  # アライアンス本部長
    10149886: 'Sales',  # 西日本営業部長
    10149889: 'Sales',  # 営業企画部長
    10149892: 'Sales',  # ソリューション営業部長
    10149898: 'Sales',  # 営業
    10149901: 'Sales',  # 営業
    9179987: 'Sales',  # 営業本部長
    9179993: 'Sales',  # 東日本営業部長
    
    # Additional Regional departments (Korea)
    1446870098522: 'Operations',  # 관리본부장 (한국)
    1446870098525: 'Sales',  # 영업본부장 (한국)
    1446870098526: 'Finance',  # 자금담당 (한국), 재무팀장 (한국), 세무담당 (한국)
    1446870098530: 'Sales',  # 영업1팀 수석, 영업1팀 주임, 영업1팀 선임
    1446870098533: 'Sales',  # 영업2팀 수석, 영업2팀장 (한국), 영업2팀 선임
    7510280: 'Procurement',  # Purchasing Operator Korea, Purchasing Manager Korea, Program Management Office Korea
    
    # Additional Service/Customer Service departments
    2190301: 'Customer Service',  # VP Shared Services, Customer Service Agent, Customer Service Agent (Ext)
    1446870398520: 'Customer Service',  # Service Manager, Dispatcher, Field Technician
    1446870398522: 'Customer Service',  # Dispatcher, Service Agent, Field Technician
    1446870398524: 'Customer Service',  # Service Manager, SVP Sales & Service Management, SVP Service Management
    
    # Additional Research & Development departments
    7181694: 'Research & Development',  # Senior Designer, VP Research and Development, Product Design Specialist
    11049764: 'Research & Development',  # Engineer II, Business Developer Senior, Business Developer Associate, Head of R&D
    
    # Additional Consulting departments
    812514: 'Consulting',  # Consulting/Analyst
    1446870368520: 'Consulting',  # Senior Consultant, Project Manager, Senior Consultant (DE)
    1446870368522: 'Consulting',  # VP Professional Services (US)
    1446870368523: 'Consulting',  # Director Delivery, Junior Consultant, Senior Consultant
    1446878830548: 'Consulting',  # Director, Project Manager, Senior Consultant
    1446878830549: 'Consulting',  # Senior Consultant, Project Manager, Director
    
    # Additional Project Management departments
    1446876846451: 'Project Management',  # Privacy Specialist, Project Team Member, Project Lead
    1446876846452: 'Project Management',  # VP Project Management
    1446879900711: 'Project Management',  # Scheduling Manager, Scheduler, Foreman
    1446879900712: 'Project Management',  # Property Manager, Facility Manager, Marketing Manager
    1446879900713: 'Project Management',  # Project Controller, CFO, President-Construction
    1446880589019: 'Project Management',  # Project Lead, Project Team Member
    
    # Additional Regional departments (China)
    4829616: 'Facilities',  # 設施經理 (also Facilities)
    4829646: 'Maintenance',  # 維護副總裁 (also Maintenance)
    
    # Additional Regional departments (Middle East)
    6721586: 'Operations',  # مدير المشروع, مدير التخطيط والجدولة, دعم إداري
    
    # Additional Regional departments (Other)
    3180002: 'Sales',  # 北海道支社長
    1446869919023: 'Engineering',  # Quality Controller, Engineering, Professional (also Engineering)
    
    # Additional Utility/Energy departments
    1446880696076: 'Utilities',  # Scheduler-T&D, Maintenance Specialist-T&D, Lineman-T&D
    1446880696079: 'Utilities',  # Maintenance Supervisor-Generation, Engineer-Generation, Scheduler-Generation
    
    # Additional Transportation departments
    1446876721668: 'Transportation',  # Transportation Planner, VP Transportation
    
    # Additional Privacy/Security departments
    
    # Additional Shared Services departments
    
    # Additional Regional departments (Belgium)
    
    # Additional Regional departments (Germany)
    1446870368519: 'Consulting',  # VP Professional Services (DE) (also Consulting)
    
    # Additional Regional departments (US)
    
    # Additional Regional departments (Ireland)
    
    # Additional Regional departments (South Africa)
    5229563: 'HR',  # Payroll Manager, VP Human Resources South Africa, HR Business Partner South Africa (also HR)
    
    # Additional Regional departments (Canada)
    5969568: 'HR',  # VP Human Resources Canada, HR Business Partner Canada, Project Manager (also HR)
    
    # Additional Regional departments (Poland)
    6780597: 'HR',  # HR Business Partner Poland (also HR)
    
    # Additional Regional departments (New Zealand)
    7039585: 'HR',  # HR Business Partner New Zealand (also HR)
    
    # Additional Regional departments (Netherlands)
    
    # Additional Regional departments (Italy)
    
    # Additional Regional departments (Argentina)
    
    # Additional Regional departments (Portugal)
    
    # Additional Regional departments (Brazil)
    
    # Additional Regional departments (Colombia)
    
    # Additional Regional departments (Switzerland)
    
    # Additional Regional departments (Czech Republic)
    
    # Additional Regional departments (Spain)
    
    # Additional Regional departments (Singapore)
    
    # Additional Regional departments (South East)
    
    # Additional Regional departments (Korea)
    
    # Additional Regional departments (Russia)
    
    # Additional Regional departments (Chile)
    
    # Additional Regional departments (Mexico)
    
    # Additional Regional departments (Asia Pacific)
    
    # Additional Regional departments (Americas)
    
    # Additional Regional departments (EMEA)
    
    # Additional Regional departments (Canada)
    
    # Additional Regional departments (South Africa)
    
    # Additional Regional departments (New Zealand)
    
    # Additional Regional departments (Poland)
    
    # Additional Regional departments (Ireland)
    
    # Additional Regional departments (Holdings)
    
    # Additional Regional departments (Banking)
    
    # Additional Regional departments (Real Estate)
    
    # Additional Regional departments (Construction)
    
    # Additional Regional departments (Korea - specific roles)
    
    # Additional Regional departments (Japan - specific roles)
    
    # Additional Regional departments (Middle East - specific roles)
    
    # Additional Regional departments (China - specific roles)
    
    # Additional Regional departments (Other countries)
    1446870398521: 'Sales',  # SVP Sales & Service Management (also Sales)
    
    # Additional Regional departments (Large numeric codes - likely regional/country specific)
    
    # Additional Regional departments (Very large numeric codes)
    
    # Additional Regional departments (Belgium)
    
    # Additional Regional departments (Germany)
    
    # Additional Regional departments (US)
    
    # Additional Regional departments (Service/Customer Service)
    
    # Additional Regional departments (Sales)
    
    # Additional Regional departments (Supply Chain)
    
    # Additional Regional departments (Transportation)
    
    # Additional Regional departments (Privacy)
    
    # Additional Regional departments (Project Management)
    
    # Additional Regional departments (Retail)
    
    # Additional Regional departments (Utilities)
    
    # Additional Regional departments (Operations)
    
    # Additional Regional departments (Engineering)
    
    # Additional Regional departments (Planning & Scheduling)
    
    # Additional Regional departments (Executive)
    
    # Additional Regional departments (HR)
    
    # Additional Regional departments (Sales)
}

# Location Code to Name Mapping
# Maps SAP SuccessFactors location codes to human-readable location names
# Defaults to Australia unless clear regional indication in job titles
LOCATION_CODE_TO_NAME = {
    # United States locations
    1: 'United States',  # US Director-Utilities, President United States, CEO BestRun Corporation
    3: 'United States',  # VP Professional Services (US), Director Operations (US), VP Human Resources USA
    10: 'United States',  # President United States
    15: 'United States',  # Senior Consultant (US)
    17: 'United States',  # HR Business Office Director USA, President BestRun Americas
    18: 'United States',  # Sales Manager-WEST, Financial Controller-West
    19: 'United States',  # Treasury Subsidiary Manager
    20: 'United States',  # Chief Information Officer (CIO)
    401: 'United States',  # President BestRun Banking, Head Digital Banking, VP Brand & Marketing
    
    # United Kingdom locations
    41: 'United Kingdom',  # President United Kingdom
    42: 'United Kingdom',  # (Default UK location)
    43: 'United Kingdom',  # HR Business Office Director UK, VP Human Resources UK
    44: 'United Kingdom',  # (Default UK location)
    
    # Germany locations
    46: 'Germany',  # President Germany, HR Director (DE), DE Director-Utilities, VP Professional Services (DE)
    1101: 'Germany',  # Director Operations (DE), Director Delivery
    1102: 'Germany',  # HRBP Sales & Services, HR Professional
    1103: 'Germany',  # Production Director (BE), HR Business Office Director (BE), President (BE) - Belgium
    
    # France locations
    47: 'France',  # (Default France location)
    48: 'France',  # (Default France location)
    49: 'France',  # President France, VP Human Resources FR, HR Business Office Director FR
    
    # Brazil locations
    51: 'Brazil',  # President Brazil, VP Human Resources Brazil, HR Business Office Director BR
    
    # Spain locations
    52: 'Spain',  # (Default Spain location)
    801: 'Spain',  # HR Business Partner Spain, President Spain, SVP Operations & Maintenance Spain
    802: 'Spain',  # Production Director
    803: 'Spain',  # Program & Management Office
    
    # Australia locations (default)
    2: 'Australia',  # Default Australia
    4: 'Australia',  # Default Australia
    5: 'Australia',  # Default Australia
    6: 'Australia',  # Default Australia
    7: 'Australia',  # Default Australia
    8: 'Australia',  # Default Australia
    9: 'Australia',  # Default Australia
    11: 'Australia',  # Default Australia
    12: 'Australia',  # Default Australia
    13: 'Australia',  # Default Australia
    14: 'Australia',  # Default Australia
    16: 'Australia',  # Default Australia
    21: 'Australia',  # Default Australia
    22: 'Australia',  # Default Australia
    23: 'Australia',  # Default Australia
    24: 'Australia',  # Default Australia
    25: 'Australia',  # Default Australia
    26: 'Australia',  # Default Australia
    27: 'Australia',  # Default Australia
    28: 'Australia',  # Default Australia
    29: 'Australia',  # Default Australia
    30: 'Australia',  # Default Australia
    31: 'Australia',  # Default Australia
    32: 'Australia',  # Default Australia
    33: 'Australia',  # Default Australia
    34: 'Australia',  # Default Australia
    35: 'Australia',  # Default Australia
    36: 'Australia',  # Default Australia
    37: 'Australia',  # Default Australia
    38: 'Australia',  # Default Australia
    39: 'Australia',  # Default Australia
    40: 'Australia',  # Default Australia
    45: 'Australia',  # Default Australia
    50: 'Australia',  # Default Australia
    53: 'Australia',  # President Australia, Managing Director Operations & Maintenance
    54: 'Australia',  # Default Australia
    55: 'Australia',  # Default Australia
    56: 'Australia',  # Default Australia
    57: 'Australia',  # Default Australia
    58: 'Australia',  # Default Australia
    59: 'Australia',  # Default Australia
    60: 'Australia',  # Default Australia
    61: 'Australia',  # Default Australia
    62: 'Australia',  # Default Australia
    63: 'Australia',  # Default Australia
    65: 'Australia',  # Default Australia
    66: 'Australia',  # Default Australia (Note: 66 also has Hong Kong, but defaulting to Australia)
    67: 'Australia',  # Default Australia (Note: 67 also has Taiwan, but defaulting to Australia)
    68: 'Australia',  # Default Australia
    69: 'Australia',  # Default Australia
    70: 'Australia',  # Default Australia
    71: 'Australia',  # Default Australia
    72: 'Australia',  # Default Australia
    73: 'Australia',  # Default Australia
    74: 'Australia',  # Default Australia
    75: 'Australia',  # Default Australia
    76: 'Australia',  # Default Australia
    77: 'Australia',  # Default Australia
    78: 'Australia',  # Default Australia
    79: 'Australia',  # Default Australia
    80: 'Australia',  # Default Australia
    81: 'Australia',  # Default Australia (Note: 81 also has Singapore, but defaulting to Australia)
    82: 'Australia',  # Default Australia
    83: 'Australia',  # Default Australia
    84: 'Australia',  # Default Australia
    85: 'Australia',  # Default Australia
    86: 'Australia',  # Default Australia
    87: 'Australia',  # Default Australia
    88: 'Australia',  # Default Australia
    89: 'Australia',  # Default Australia
    90: 'Australia',  # Default Australia
    91: 'Australia',  # Default Australia
    92: 'Australia',  # Default Australia
    93: 'Australia',  # Default Australia
    94: 'Australia',  # Default Australia
    95: 'Australia',  # Default Australia
    96: 'Australia',  # Default Australia
    97: 'Australia',  # Default Australia
    98: 'Australia',  # Default Australia
    99: 'Australia',  # Default Australia
    100: 'Australia',  # Default Australia
    
    # Japan locations
    64: 'Japan',  # Japanese characters: 営業, 北海道支社長, ソリューション営業部長, etc.
    1060: 'Japan',  # 営業
    
    # Hong Kong locations
    66: 'Hong Kong',  # SVP Hong Kong Operations & Maintenance, HR Business Partner - HKG
    
    # Taiwan/China locations
    7: 'China',  # President China, HR Business Office Director CN, 管理员, 计划和排班经理
    67: 'Taiwan',  # Chinese characters: 釆購副總裁, 中国台湾地区维护和运营高级副总裁, 維護副總裁, 運營副總裁, 設施經理
    
    # Singapore locations
    81: 'Singapore',  # President Singapore, President SEA, HR Business Partner SGP
    1300: 'Singapore',  # President SE, VP Human Resources SE
    1301: 'Singapore',  # HR Business Office Director, HR Business Partner
    1302: 'Singapore',  # Digital Expert, Digital Consultant
    1303: 'Singapore',  # Planner, Scheduler
    1304: 'Singapore',  # Assembly Worker, Senior Production Technician
    
    # South Africa locations
    201: 'South Africa',  # President Best Run South Africa, VP Human Resources South Africa, HR Business Partner South Africa
    202: 'South Africa',  # HR Business Partner South Africa, Recruiting Manager
    203: 'South Africa',  # Facilities Manager, Executive Assistant to the President
    
    # Canada locations
    261: 'Canada',  # Project Execution Lead, Recruiting Manager
    262: 'Canada',  # HR Business Partner Canada
    263: 'Canada',  # Executive Assistant to the President, SVP Operations & Maintenance
    264: 'Canada',  # Project Manager, VP Human Resources Canada
    265: 'Canada',  # Production Director
    266: 'Canada',  # Director of Talent Management, President Best Run Canada
    
    # Mexico locations
    281: 'Mexico',  # President Best Run Mexico, HR Business Partner Mexico, SVP Operations & Maintenance
    
    # Qatar locations
    341: 'Qatar',  # Arabic: رئيس قطر, شريك أعمال الموارد البشرية قطر, etc.
    
    # Saudi Arabia locations
    361: 'Saudi Arabia',  # Arabic: رئيس المملكة العربية السعودية, شريك أعمال الموارد البشرية السعودية
    
    # Poland locations
    381: 'Poland',  # President BestRun Poland, HR Business Partner Poland
    
    # UAE locations
    421: 'UAE',  # Arabic: رئيس الامارات المتحدة العربية, الموارد البشرية الإمارات العربية المتحدة شريك الأعمال في
    
    # New Zealand locations
    441: 'New Zealand',  # President Best Run New Zealand, HR Business Partner New Zealand
    
    # Korea locations
    541: 'Korea',  # Korean: 영업1팀, 관리본부장, 베스트런한국 사장, etc.
    
    # Netherlands locations
    561: 'Netherlands',  # VP Human Resources NL, HR Business Office Director
    562: 'Netherlands',  # HR Business Partner, Payroll Admin
    
    # Italy locations
    581: 'Italy',  # President BestRun Italy, HR Business Partner Italy
    
    # Argentina locations
    601: 'Argentina',  # President BestRun Argentina, HR Business Partner Argentina
    
    # Portugal locations
    641: 'Portugal',  # VP Human Resources PT, Planning & Scheduling Manager
    642: 'Portugal',  # President Portugal, HR Business Partner
    
    # Chile locations
    661: 'Chile',  # President Chile, HR Business Partner Chile
    
    # Colombia locations
    681: 'Colombia',  # President Colombia, HR Business Partner Colombia
    
    # Other South America
    701: 'Colombia',  # VP Operations, Production Director
    
    # Switzerland locations
    721: 'Switzerland',  # President Switzerland, HR Business Partner CHE, SVP Operations and Maintenance CHE
    742: 'Switzerland',  # Payroll Manager
    
    # Czech Republic locations
    761: 'Czech Republic',  # President Czech Republic, HR Business Partner CZE, SVP Operations and Maintenance CZE
    762: 'Czech Republic',  # Production Director CZE, VP Operations CZE, Program Management Office CZE
    
    # Ireland locations
    881: 'Ireland',  # President Best Run Ireland, HR Business Partner Ireland, Operations Manager
    
    # Retail locations (defaulting to Australia)
    901: 'Australia',  # Retail locations
    902: 'Australia',  # Retail locations
    907: 'Australia',  # Retail locations
    
    # Other large numeric codes (defaulting to Australia)
    821: 'Australia',  # CTO, Inside Sales, VP Sales, IT Manager
    822: 'Australia',  # Business Developer, Maintenance Manager, Production Manager
    823: 'Australia',  # Market Researcher, Data Scientist, Marketing Manager
    841: 'Australia',  # Account Executive, Service Agent
    842: 'Australia',  # Account Executive
    861: 'Australia',  # Account Executive
    1103: 'Belgium',  # Production Director (BE), HR Business Office Director (BE), President (BE)
}

