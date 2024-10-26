-- 1. What is the distribution of employee ages across different satisfaction levels?
SELECT e.Age, s.SatisfactionLevel,
   COUNT(*) as EmployeeCount
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN SatisfiedLevel s ON pr.JobSatisfaction = s.SatisfactionID
   GROUP BY e.Age, s.SatisfactionLevel, s.SatisfactionID
   ORDER BY e.Age, s.SatisfactionID;

-- 2. How many employees have a specific education level and receive a specific manager rating?
SELECT el.EducationLevel, rl.RatingLevel,
   COUNT(*) as EmployeeCount
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN EducationLevel el ON e.Education = el.EducationLevelID
JOIN RatingLevel rl ON pr.ManagerRating = rl.RatingID
   GROUP BY el.EducationLevel, rl.RatingLevel, el.EducationLevelID, rl.RatingID
   ORDER BY el.EducationLevelID, rl.RatingID;

-- 3. Factors influencing Attrition
SELECT 
    e.Age,
    e.YearsAtCompany,
    e.DistanceFromHome_KM,
    e.OverTime,
    pr.WorkLifeBalance,
    e.Attrition
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN SatisfiedLevel s ON pr.JobSatisfaction = s.SatisfactionID;

-- 4. Are there any patterns or correlations between the different satisfaction levels within each department?
SELECT e.Department,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.EnvironmentSatisfaction AS FLOAT)) as AvgEnvironmentSatisfaction,
    AVG(CAST(pr.RelationshipSatisfaction AS FLOAT)) as AvgRelationshipSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
   GROUP BY e.Department;

-- 5. Is there a positive correlation between taking more training opportunities and higher levels of job satisfaction?
SELECT pr.TrainingOpportunitiesTaken,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM PerformanceRating pr
   GROUP BY pr.TrainingOpportunitiesTaken
   ORDER BY pr.TrainingOpportunitiesTaken;


-- 6. Education Level vs Job Satisfaction
SELECT el.EducationLevel, s.SatisfactionLevel,
   COUNT(*) as EmployeeCount
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN EducationLevel el ON e.Education = el.EducationLevelID
JOIN SatisfiedLevel s ON pr.JobSatisfaction = s.SatisfactionID
   GROUP BY el.EducationLevel, s.SatisfactionLevel, el.EducationLevelID, s.SatisfactionID
   ORDER BY el.EducationLevelID, s.SatisfactionID;

-- 7. Gender vs Satisfaction Levels in Different Departments
SELECT e.Department, e.Gender, s.SatisfactionLevel,
   COUNT(*) as EmployeeCount
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN SatisfiedLevel s ON pr.JobSatisfaction = s.SatisfactionID
   GROUP BY e.Department, e.Gender, s.SatisfactionLevel, s.SatisfactionID
   ORDER BY e.Department, e.Gender, s.SatisfactionID;

-- 8. Ethnicity vs Satisfaction Metrics
SELECT e.Ethnicity, 
       AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
       AVG(CAST(pr.EnvironmentSatisfaction AS FLOAT)) as AvgEnvironmentSatisfaction,
       AVG(CAST(pr.RelationshipSatisfaction AS FLOAT)) as AvgRelationshipSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
   GROUP BY e.Ethnicity;

-- 9. Distance from Home vs Job Satisfaction
SELECT 
    CASE 
        WHEN e.DistanceFromHome_KM <= 5 THEN '0-5 km'
        WHEN e.DistanceFromHome_KM <= 10 THEN '6-10 km'
        WHEN e.DistanceFromHome_KM <= 20 THEN '11-20 km'
        ELSE '20+ km'
    END AS DistanceRange,
     AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY 
    CASE 
        WHEN e.DistanceFromHome_KM <= 5 THEN '0-5 km'
        WHEN e.DistanceFromHome_KM <= 10 THEN '6-10 km'
        WHEN e.DistanceFromHome_KM <= 20 THEN '11-20 km'
        ELSE '20+ km'
    END
  ORDER BY DistanceRange;

-- 10. Years at Company vs Performance Ratings
SELECT 
    CASE 
        WHEN e.YearsAtCompany <= 2 THEN '0-2 years'
        WHEN e.YearsAtCompany <= 5 THEN '3-5 years'
        WHEN e.YearsAtCompany <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END AS YearsAtCompanyRange,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY 
    CASE 
        WHEN e.YearsAtCompany <= 2 THEN '0-2 years'
        WHEN e.YearsAtCompany <= 5 THEN '3-5 years'
        WHEN e.YearsAtCompany <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END
  ORDER BY YearsAtCompanyRange;

-- 11. Work-Life Balance vs Performance Ratings
SELECT pr.WorkLifeBalance,
   AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM PerformanceRating pr
   GROUP BY pr.WorkLifeBalance
   ORDER BY pr.WorkLifeBalance;

-- 12. Years Since Last Promotion vs Job Satisfaction
SELECT 
    CASE 
        WHEN e.YearsSinceLastPromotion = 0 THEN 'This year'
        WHEN e.YearsSinceLastPromotion = 1 THEN '1 year'
        WHEN e.YearsSinceLastPromotion <= 3 THEN '2-3 years'
        ELSE '4+ years'
    END AS YearsSincePromotionRange,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY 
    CASE 
        WHEN e.YearsSinceLastPromotion = 0 THEN 'This year'
        WHEN e.YearsSinceLastPromotion = 1 THEN '1 year'
        WHEN e.YearsSinceLastPromotion <= 3 THEN '2-3 years'
        ELSE '4+ years'
    END
  ORDER BY YearsSincePromotionRange;

-- 13. Salary vs Job Satisfaction Across Education Levels
SELECT el.EducationLevel,
    CASE 
        WHEN e.Salary <= 50000 THEN 'Up to 50k'
        WHEN e.Salary <= 100000 THEN '50k-100k'
        WHEN e.Salary <= 150000 THEN '100k-150k'
        ELSE '150k+'
    END AS SalaryRange,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN EducationLevel el ON e.Education = el.EducationLevelID
  GROUP BY 
    el.EducationLevel,
    el.EducationLevelID,
    CASE 
        WHEN e.Salary <= 50000 THEN 'Up to 50k'
        WHEN e.Salary <= 100000 THEN '50k-100k'
        WHEN e.Salary <= 150000 THEN '100k-150k'
        ELSE '150k+'
    END
  ORDER BY el.EducationLevelID, SalaryRange;

-- 14. Stock Option Levels vs Overall Satisfaction
SELECT e.StockOptionLevel,
   AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.StockOptionLevel
  ORDER BY e.StockOptionLevel;

-- 15. Overtime vs Job Satisfaction and Performance Ratings
SELECT e.OverTime, 
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.OverTime;


-- 16. Factors Correlated with Attrition
SELECT 
    AVG(CASE WHEN e.Attrition = 1 THEN 1.0 ELSE 0.0 END) as AttritionRate,
    AVG(CAST(e.Age AS FLOAT)) as AvgAge,
    AVG(CAST(e.YearsAtCompany AS FLOAT)) as AvgYearsAtCompany,
    AVG(CAST(e.DistanceFromHome_KM AS FLOAT)) as AvgDistanceFromHome,
    AVG(CAST(e.Salary AS FLOAT)) as AvgSalary,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.WorkLifeBalance AS FLOAT)) as AvgWorkLifeBalance
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID;

-- 17. Job Satisfaction vs Attrition
SELECT s.SatisfactionLevel,
    COUNT(*) as TotalEmployees,
    SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) as AttritionCount,
    CAST(SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as AttritionRate
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
JOIN SatisfiedLevel s ON pr.JobSatisfaction = s.SatisfactionID
  GROUP BY s.SatisfactionLevel, s.SatisfactionID
  ORDER BY s.SatisfactionID;

-- 18. Years with Current Manager vs Attrition Rates
SELECT e.YearsWithCurrManager,
    COUNT(*) as TotalEmployees,
    SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) as AttritionCount,
    CAST(SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as AttritionRate
FROM Employee e
   GROUP BY e.YearsWithCurrManager
   ORDER BY e.YearsWithCurrManager;

-- 19. Satisfaction Metrics Impact on Attrition
SELECT 
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.RelationshipSatisfaction AS FLOAT)) as AvgRelationshipSatisfaction,
    AVG(CAST(pr.EnvironmentSatisfaction AS FLOAT)) as AvgEnvironmentSatisfaction,
    SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) as AttritionCount,
    COUNT(*) as TotalEmployees,
    CAST(SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as AttritionRate
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID;

-- 20. Job Role Satisfaction Variation
SELECT e.JobRole,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    MIN(CAST(pr.JobSatisfaction AS FLOAT)) as MinJobSatisfaction,
    MAX(CAST(pr.JobSatisfaction AS FLOAT)) as MaxJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
   GROUP BY e.JobRole
   ORDER BY AvgJobSatisfaction DESC;

-- 21. Job Role and Performance Ratings
SELECT e.JobRole,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
   GROUP BY e.JobRole
   ORDER BY AvgPerformanceRating DESC;

-- 22. Training Opportunities and Job Satisfaction
SELECT pr.TrainingOpportunitiesWithinYear,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM PerformanceRating pr
  GROUP BY pr.TrainingOpportunitiesWithinYear
  ORDER BY pr.TrainingOpportunitiesWithinYear;

-- 23. Education Field and Job Satisfaction within Departments
SELECT e.Department, e.EducationField,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.Department, e.EducationField
  ORDER BY e.Department, AvgJobSatisfaction DESC;

-- 24. Work-Life Balance and Satisfaction Metrics
SELECT pr.WorkLifeBalance,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.EnvironmentSatisfaction AS FLOAT)) as AvgEnvironmentSatisfaction,
    AVG(CAST(pr.RelationshipSatisfaction AS FLOAT)) as AvgRelationshipSatisfaction
FROM PerformanceRating pr
  GROUP BY pr.WorkLifeBalance
  ORDER BY pr.WorkLifeBalance;

-- 25. Business Travel and Work-Life Balance Satisfaction
SELECT e.BusinessTravel,
    AVG(CAST(pr.WorkLifeBalance AS FLOAT)) as AvgWorkLifeBalance
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.BusinessTravel;

-- 26. Self-Rating vs Manager Rating and Job Satisfaction
SELECT 
    CASE 
        WHEN pr.SelfRating > pr.ManagerRating THEN 'Higher Self-Rating'
        WHEN pr.SelfRating < pr.ManagerRating THEN 'Lower Self-Rating'
        ELSE 'Equal Rating'
    END AS RatingComparison,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM PerformanceRating pr
GROUP BY 
    CASE 
        WHEN pr.SelfRating > pr.ManagerRating THEN 'Higher Self-Rating'
        WHEN pr.SelfRating < pr.ManagerRating THEN 'Lower Self-Rating'
        ELSE 'Equal Rating'
    END;

-- 27. Years with Current Manager and Satisfaction Metrics
SELECT e.YearsWithCurrManager,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.RelationshipSatisfaction AS FLOAT)) as AvgRelationshipSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.YearsWithCurrManager
  ORDER BY e.YearsWithCurrManager;

-- 28. Marital Status, Work-Life Balance, and Job Satisfaction
SELECT e.MaritalStatus,
    AVG(CAST(pr.WorkLifeBalance AS FLOAT)) as AvgWorkLifeBalance,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.MaritalStatus;


-- 30. Performance Rating Trends Over Time
SELECT 
    YEAR(pr.ReviewDate) as ReviewYear,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM PerformanceRating pr
  GROUP BY YEAR(pr.ReviewDate)
  ORDER BY ReviewYear;

-- 31. Years in Current Role vs Performance Ratings
SELECT e.YearsInMostRecentRole,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.YearsInMostRecentRole
  ORDER BY e.YearsInMostRecentRole;


-- 32. Department and Gender Distribution
SELECT e.Department, e.Gender,
    COUNT(*) as EmployeeCount,
    CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER (PARTITION BY e.Department) as Percentage
FROM Employee e
  GROUP BY e.Department, e.Gender
  ORDER BY e.Department, e.Gender;


-- 33. Age Distribution by Department
SELECT e.Department,
    AVG(e.Age) as AvgAge,
    MIN(e.Age) as MinAge,
    MAX(e.Age) as MaxAge
FROM Employee e
  GROUP BY e.Department
  ORDER BY AvgAge DESC;

-- 34. Correlation between Training Opportunities Taken and Years at Company
SELECT 
    CASE 
        WHEN e.YearsAtCompany <= 2 THEN '0-2 years'
        WHEN e.YearsAtCompany <= 5 THEN '3-5 years'
        WHEN e.YearsAtCompany <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END AS YearsAtCompanyRange,
    AVG(CAST(pr.TrainingOpportunitiesTaken AS FLOAT)) as AvgTrainingsTaken
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
GROUP BY 
    CASE 
        WHEN e.YearsAtCompany <= 2 THEN '0-2 years'
        WHEN e.YearsAtCompany <= 5 THEN '3-5 years'
        WHEN e.YearsAtCompany <= 10 THEN '6-10 years'
        ELSE '10+ years'
    END
ORDER BY YearsAtCompanyRange;

-- 35. Job Role Progression (Assuming higher job levels indicate progression)
SELECT e.JobRole,
    AVG(e.YearsAtCompany) as AvgYearsAtCompany,
    AVG(e.YearsSinceLastPromotion) as AvgYearsSinceLastPromotion,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.JobRole
  ORDER BY AvgPerformanceRating DESC;


-- 36. Impact of Distance from Home on Overtime
SELECT 
    CASE 
        WHEN e.DistanceFromHome_KM <= 5 THEN '0-5 km'
        WHEN e.DistanceFromHome_KM <= 10 THEN '6-10 km'
        WHEN e.DistanceFromHome_KM <= 20 THEN '11-20 km'
        ELSE '20+ km'
    END AS DistanceRange,
    SUM(CASE WHEN e.OverTime = 1 THEN 1 ELSE 0 END) as OvertimeCount,
    COUNT(*) as TotalEmployees,
    CAST(SUM(CASE WHEN e.OverTime = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as OvertimeRate
FROM Employee e
GROUP BY 
    CASE 
        WHEN e.DistanceFromHome_KM <= 5 THEN '0-5 km'
        WHEN e.DistanceFromHome_KM <= 10 THEN '6-10 km'
        WHEN e.DistanceFromHome_KM <= 20 THEN '11-20 km'
        ELSE '20+ km'
    END
ORDER BY DistanceRange;


-- 37. Relationship between Stock Option Level and Years at Company
SELECT e.StockOptionLevel,
    AVG(e.YearsAtCompany) as AvgYearsAtCompany,
    MIN(e.YearsAtCompany) as MinYearsAtCompany,
    MAX(e.YearsAtCompany) as MaxYearsAtCompany
FROM Employee e
  GROUP BY e.StockOptionLevel
  ORDER BY e.StockOptionLevel;


-- 38. Impact of Education Field on Salary within Job Roles
SELECT e.JobRole, e.EducationField,
    AVG(e.Salary) as AvgSalary,
    MIN(e.Salary) as MinSalary,
    MAX(e.Salary) as MaxSalary
FROM Employee e
  GROUP BY e.JobRole, e.EducationField
  ORDER BY e.JobRole, AvgSalary DESC;


-- 39. Relationship between Business Travel and Years Since Last Promotion
SELECT e.BusinessTravel,
    AVG(e.YearsSinceLastPromotion) as AvgYearsSincePromotion,
    MIN(e.YearsSinceLastPromotion) as MinYearsSincePromotion,
    MAX(e.YearsSinceLastPromotion) as MaxYearsSincePromotion
FROM Employee e
  GROUP BY e.BusinessTravel;


-- 40. Correlation between Environment Satisfaction and Years with Current Manager
SELECT e.YearsWithCurrManager,
    AVG(CAST(pr.EnvironmentSatisfaction AS FLOAT)) as AvgEnvironmentSatisfaction
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.YearsWithCurrManager
  ORDER BY e.YearsWithCurrManager;


-- 41. Impact of Marital Status on Attrition Rates
SELECT e.MaritalStatus,
    COUNT(*) as TotalEmployees,
    SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) as AttritionCount,
    CAST(SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as AttritionRate
FROM Employee e
  GROUP BY e.MaritalStatus;

-- 42. Relationship between Performance Rating and Years Since Last Promotion
SELECT 
    CASE 
        WHEN e.YearsSinceLastPromotion = 0 THEN 'This year'
        WHEN e.YearsSinceLastPromotion = 1 THEN '1 year'
        WHEN e.YearsSinceLastPromotion <= 3 THEN '2-3 years'
        ELSE '4+ years'
    END AS YearsSincePromotionRange,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
GROUP BY 
    CASE 
        WHEN e.YearsSinceLastPromotion = 0 THEN 'This year'
        WHEN e.YearsSinceLastPromotion = 1 THEN '1 year'
        WHEN e.YearsSinceLastPromotion <= 3 THEN '2-3 years'
        ELSE '4+ years'
    END
ORDER BY YearsSincePromotionRange;

-- 43. Department-wise Training Opportunities Utilization
SELECT e.Department,
    AVG(CAST(pr.TrainingOpportunitiesWithinYear AS FLOAT)) as AvgOpportunitiesOffered,
    AVG(CAST(pr.TrainingOpportunitiesTaken AS FLOAT)) as AvgOpportunitiesTaken,
    AVG(CAST(pr.TrainingOpportunitiesTaken AS FLOAT)) / AVG(CAST(pr.TrainingOpportunitiesWithinYear AS FLOAT)) as UtilizationRate
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.Department
  ORDER BY UtilizationRate DESC;

-- 44. Impact of State on Various Metrics
SELECT e.State,
    AVG(e.Salary) as AvgSalary,
    AVG(CAST(pr.JobSatisfaction AS FLOAT)) as AvgJobSatisfaction,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgPerformanceRating,
    CAST(SUM(CASE WHEN e.Attrition = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as AttritionRate
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY e.State
  ORDER BY AvgSalary DESC;

-- 45. Relationship between Hire Date and Current Performance
SELECT 
    YEAR(e.HireDate) as HireYear,
    AVG(CAST(pr.ManagerRating AS FLOAT)) as AvgCurrentPerformance,
    COUNT(*) as EmployeeCount
FROM Employee e
JOIN PerformanceRating pr ON e.EmployeeID = pr.EmployeeID
  GROUP BY YEAR(e.HireDate)
  ORDER BY HireYear;


-- 46. What is the distribution of employees across different education levels? 
-- How does the education level correlate with job roles, salaries, and performance ratings?
SELECT EducationField,
   COUNT(*) AS num_employees 
FROM Employee 
   GROUP BY EducationField;
SELECT e.EducationField, e.JobRole,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating) AS avg_manager_rating,
   AVG(e.Salary) AS avg_salary 
FROM Employee e 
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID 
   GROUP BY e.EducationField, e.JobRole;

-- 47. Analyze the relationship between years of experience:
-- (YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion, YearsWithCurrManager)
-- and performance ratings (SelfRating, ManagerRating).
-- Does more experience lead to better performance?
 SELECT AVG(YearsAtCompany) AS avg_years_at_company,
       AVG(YearsInMostRecentRole) AS avg_years_in_role,
       AVG(YearsSinceLastPromotion) AS avg_years_since_promotion,
	   AVG(YearsWithCurrManager) AS avg_years_with_manager,
       AVG(p.SelfRating) AS avg_self_rating,
	   AVG(p.ManagerRating) AS avg_manager_rating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.YearsAtCompany, e.YearsInMostRecentRole, e.YearsSinceLastPromotion, e.YearsWithCurrManager;

-- 48. Investigate the impact of job satisfaction factors (EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction)
-- on employee attrition (Attrition column). 
-- Are there specific factors that contribute more to employee turnover?
SELECT p.EnvironmentSatisfaction, p.JobSatisfaction, p.RelationshipSatisfaction,
   AVG(CAST(e.Attrition AS FLOAT)) AS avg_attrition 
FROM Employee e 
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID 
   GROUP BY p.EnvironmentSatisfaction, p.JobSatisfaction, p.RelationshipSatisfaction;

-- 49. Explore the relationship between training opportunities:
-- (TrainingOpportunitiesWithinYear, TrainingOpportunitiesTaken) and performance ratings.
-- Does providing more training opportunities lead to better performance?
SELECT p.TrainingOpportunitiesWithinYear, p.TrainingOpportunitiesTaken,
       AVG(p.SelfRating) AS avg_self_rating,
	   AVG(p.ManagerRating) AS avg_manager_rating
FROM PerformanceRating p
   GROUP BY p.TrainingOpportunitiesWithinYear, p.TrainingOpportunitiesTaken;

-- 50. Analyze the distribution of employees across different departments 
-- and examine the correlation between department and factors like salary, job satisfaction, and performance ratings.
SELECT e.Department, 
   COUNT(*) AS num_employees,
   AVG(e.Salary) AS avg_salary,
   AVG(p.JobSatisfaction) AS avg_job_satisfaction,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating) AS avg_manager_rating 
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID 
   GROUP BY e.Department;

-- 51. Investigate the impact of work-life balance on job satisfaction and performance ratings.
-- Are employees with better work-life balance more satisfied and perform better?
SELECT p.WorkLifeBalance,
    AVG(p.JobSatisfaction) AS AvgJobSatisfaction,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM PerformanceRating p
JOIN Employee e ON e.EmployeeID = p.EmployeeID
   GROUP BY p.WorkLifeBalance;

-- 52. Examine the relationship between distance from home (DistanceFromHome_KM) and job satisfaction factors.
-- Does commute time affect employee satisfaction and performance?
SELECT e.DistanceFromHome_KM,
   AVG(p.EnvironmentSatisfaction) AS avg_env_satisfaction,
   AVG(p.JobSatisfaction) AS avg_job_satisfaction,
   AVG(p.RelationshipSatisfaction) AS avg_rel_satisfaction,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating) AS avg_manager_rating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID 
   GROUP BY e.DistanceFromHome_KM;

-- 53. Analyze the distribution of performance ratings across different managers (using ManagerRating and EmployeeID columns).
-- Are there managers who consistently rate their employees higher or lower than others?
SELECT p.ManagerRating,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating) AS avg_manager_rating
FROM PerformanceRating p
JOIN Employee e ON e.EmployeeID = p.EmployeeID
   GROUP BY p.ManagerRating;

-- 54. Investigate the relationship between stock option level (StockOptionLevel) and employee attrition.
-- Do higher stock option levels lead to lower attrition rates?
SELECT e.StockOptionLevel,
   AVG(CAST(e.Attrition AS FLOAT)) AS avg_attrition
FROM Employee e 
   GROUP BY e.StockOptionLevel;

-- 55. Explore the correlation between overtime (Overtime column) and performance ratings.
-- Does working overtime have a positive or negative impact on performance?
SELECT e.Overtime,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating) AS avg_manager_rating
FROM Employee e  
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.Overtime;

-- 56. Analyze the distribution of employees across different ethnicities and genders,
-- and examine any potential disparities in salary, job roles, or performance ratings.
SELECT e.Gender, e.Ethnicity,
    COUNT(*) AS num_employees,
    AVG(e.Salary) AS avg_salary,
    AVG(p.SelfRating) AS avg_self_rating,
	AVG(p.ManagerRating) AS avg_manager_rating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.Gender, e.Ethnicity;

-- 57. Investigate the relationship between business travel (BusinessTravel column) and job satisfaction factors.
-- Does frequent travel affect employee satisfaction and performance?
SELECT e.BusinessTravel,
   AVG(p.EnvironmentSatisfaction) AS avg_env_satisfaction,
   AVG(p.JobSatisfaction) AS avg_job_satisfaction,
   AVG(p.RelationshipSatisfaction) AS avg_rel_satisfaction,
   AVG(p.SelfRating) AS avg_self_rating,
   AVG(p.ManagerRating)AS avg_manager_rating
FROM Employee e 
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID 
   GROUP BY e.BusinessTravel;

-- 58. What is the average salary and performance rating for employees based on their age groups?
SELECT 
    CASE
        WHEN Age BETWEEN 18 AND 30 THEN '18-30'
        WHEN Age BETWEEN 31 AND 40 THEN '31-40'
        WHEN Age BETWEEN 41 AND 50 THEN '41-50'
        WHEN Age > 50 THEN '51+'
    END AS AgeGroup,
    AVG(Salary) AS AvgSalary,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY Age;

-- 59. How does the average performance rating vary across different job roles and departments?
SELECT 
    e.JobRole,
    e.Department,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.JobRole, e.Department;

-- 60. What is the relationship between the number of years an employee has been with the company 
--and their job satisfaction, attrition rate, and performance ratings?
SELECT 
    YearsAtCompany,
    AVG(p.JobSatisfaction) AS AvgJobSatisfaction,
    AVG(CAST(e.Attrition AS FLOAT)) AS AvgAttritionRate,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY YearsAtCompany;

-- 61. How does the average performance rating vary based on the level of education and job role?
SELECT e.EducationField, e.JobRole,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.EducationField, e.JobRole;

-- 62. What is the relationship between the number of training programs attended 
--and employee performance ratings, job satisfaction, and attrition rate?
SELECT P.TrainingOpportunitiesTaken,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating,
    AVG(p.JobSatisfaction) AS AvgJobSatisfaction,
    AVG(CAST(e.Attrition AS FLOAT)) AS AvgAttritionRate
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY P.TrainingOpportunitiesTaken;

-- 63. How does the average salary and performance rating vary across different marital statuses and gender?
SELECT e.MaritalStatus, e.Gender,
    AVG(e.Salary) AS AvgSalary,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.MaritalStatus, e.Gender;

-- 64. What is the relationship between the number of years at company an employee has worked for
-- and their performance ratings, job satisfaction, and environment satisfaction?
SELECT e.YearsAtCompany,
    AVG(p.SelfRating) AS AvgSelfRating,
    AVG(p.ManagerRating) AS AvgManagerRating,
    AVG(p.JobSatisfaction) AS AvgJobSatisfaction,
    AVG(p.EnvironmentSatisfaction) AS AvgEnvironmentSatisfaction
FROM Employee e
JOIN PerformanceRating p ON e.EmployeeID = p.EmployeeID
   GROUP BY e.YearsAtCompany;