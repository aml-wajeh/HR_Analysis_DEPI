                 -- Procedure to clean, validate and preprocess Employee table --


CREATE OR ALTER PROCEDURE CleanEmployeeData
AS
BEGIN
    -- Remove duplicates -- 

    WITH CTE AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY EmployeeID ORDER BY (SELECT NULL)) AS RowNum
        FROM Employee
    )
    DELETE FROM CTE WHERE RowNum > 1;

    -- Remove invalid records --

    DELETE FROM Employee
    WHERE Age < 18 OR Age > 70
       OR Salary < 0
       OR YearsAtCompany < 0
       OR YearsInMostRecentRole < 0
       OR YearsSinceLastPromotion < 0
       OR YearsWithCurrManager < 0;

    -- Handle missing values --

    UPDATE Employee
    SET Age = COALESCE(Age, (SELECT AVG(Age) FROM Employee)),
        DistanceFromHome_KM = COALESCE(DistanceFromHome_KM, (SELECT AVG(DistanceFromHome_KM) FROM Employee)),
        Salary = COALESCE(Salary, (SELECT AVG(Salary) FROM Employee WHERE Department = e.Department)),
        YearsAtCompany = COALESCE(YearsAtCompany, 0),
        YearsInMostRecentRole = COALESCE(YearsInMostRecentRole, 0),
        YearsSinceLastPromotion = COALESCE(YearsSinceLastPromotion, 0),
        YearsWithCurrManager = COALESCE(YearsWithCurrManager, 0)
    FROM Employee e
    WHERE Age IS NULL 
       OR DistanceFromHome_KM IS NULL 
       OR Salary IS NULL
       OR YearsAtCompany IS NULL
       OR YearsInMostRecentRole IS NULL
       OR YearsSinceLastPromotion IS NULL
       OR YearsWithCurrManager IS NULL;

    -- Standardize text fields -- 

    UPDATE Employee
    SET Gender = UPPER(LEFT(Gender, 1)),
        BusinessTravel = UPPER(LEFT(BusinessTravel, 1)) + LOWER(SUBSTRING(BusinessTravel, 2, LEN(BusinessTravel))),
        Department = UPPER(LEFT(Department, 1)) + LOWER(SUBSTRING(Department, 2, LEN(Department))),
        EducationField = UPPER(LEFT(EducationField, 1)) + LOWER(SUBSTRING(EducationField, 2, LEN(EducationField))),
        JobRole = UPPER(LEFT(JobRole, 1)) + LOWER(SUBSTRING(JobRole, 2, LEN(JobRole))),
        MaritalStatus = UPPER(LEFT(MaritalStatus, 1)) + LOWER(SUBSTRING(MaritalStatus, 2, LEN(MaritalStatus)));

END;


                                   -- Add Some Features --

-- Add TenureYears column (calculate the number of days between the hire date and the current date, then dividing by 365.25 to get years )
   SELECT 
    EmployeeID, 
    HireDate,
    DATEDIFF(day, HireDate, GETDATE()) / 365.25 AS TenureYears
   FROM 
    Employee;

-- Add AgeGroup (using a CASE statement to bucket ages into groups )
 SELECT
    EmployeeID,
    Age,
    CASE 
        WHEN Age BETWEEN 18 AND 25 THEN '18-25'
        WHEN Age BETWEEN 26 AND 35 THEN '26-35'
        WHEN Age BETWEEN 36 AND 45 THEN '36-45'
        WHEN Age BETWEEN 46 AND 55 THEN '46-55'
        ELSE '55+'
    END AS AgeGroup
 FROM 
    Employee;

-- Create a feature for job level based on salary ( SalaryQuartile ) column
	WITH SalaryQuartiles AS (
SELECT 
        EmployeeID, 
        Salary,
        NTILE(5) OVER (ORDER BY Salary) AS SalaryQuartile
    FROM 
        Employee
)
SELECT
    EmployeeID,
    Salary,
    CASE SalaryQuartile
        WHEN 1 THEN 'Entry'
        WHEN 2 THEN 'Junior'
        WHEN 3 THEN 'Mid'
        WHEN 4 THEN 'Senior'
        WHEN 5 THEN 'Executive'
    END AS JobLevel
FROM 
    SalaryQuartiles;




             -- Procedure to clean, validate and preprocess PerformanceRating table --

CREATE OR ALTER PROCEDURE CleanPerformanceRatingData
AS
BEGIN
      -- Remove invalid records --

    DELETE FROM PerformanceRating
    WHERE ReviewDate > GETDATE()
       OR EnvironmentSatisfaction NOT BETWEEN 1 AND 5
       OR JobSatisfaction NOT BETWEEN 1 AND 5
       OR RelationshipSatisfaction NOT BETWEEN 1 AND 5
       OR WorkLifeBalance NOT BETWEEN 1 AND 5
       OR SelfRating NOT BETWEEN 1 AND 5
       OR ManagerRating NOT BETWEEN 1 AND 5;

      -- Handle missing values --

    UPDATE PerformanceRating
    SET EnvironmentSatisfaction = COALESCE(EnvironmentSatisfaction, 3),
        JobSatisfaction = COALESCE(JobSatisfaction, 3),
        RelationshipSatisfaction = COALESCE(RelationshipSatisfaction, 3),
        WorkLifeBalance = COALESCE(WorkLifeBalance, 3),
        SelfRating = COALESCE(SelfRating, 3),
        ManagerRating = COALESCE(ManagerRating, 3),
        TrainingOpportunitiesWithinYear = COALESCE(TrainingOpportunitiesWithinYear, 0),
        TrainingOpportunitiesTaken = COALESCE(TrainingOpportunitiesTaken, 0);

      -- Ensure TrainingOpportunitiesTaken <= TrainingOpportunitiesWithinYear -- 

    UPDATE PerformanceRating
    SET TrainingOpportunitiesTaken = TrainingOpportunitiesWithinYear
    WHERE TrainingOpportunitiesTaken > TrainingOpportunitiesWithinYear;
END;


                             -- Execute the procedures --
EXEC CleanEmployeeData;
EXEC CleanPerformanceRatingData;





                                        -- Making Summary Statistics --

     -- View for detailed Employee summary statistics --

CREATE OR ALTER VIEW EmployeeSummaryStats AS
SELECT 
    COUNT(*) AS TotalEmployees,
    SUM(CASE WHEN Gender = 'M' THEN 1 ELSE 0 END) AS MaleCount,
    SUM(CASE WHEN Gender = 'F' THEN 1 ELSE 0 END) AS FemaleCount,
    AVG(Age) AS AvgAge,
    STDEV(Age) AS StdDevAge,
    MIN(Age) AS MinAge,
    MAX(Age) AS MaxAge,
    AVG(Salary) AS AvgSalary,
    STDEV(Salary) AS StdDevSalary,
    MIN(Salary) AS MinSalary,
    MAX(Salary) AS MaxSalary,
    AVG(YearsAtCompany) AS AvgYearsAtCompany,
    AVG(YearsInMostRecentRole) AS AvgYearsInMostRecentRole,
    AVG(YearsSinceLastPromotion) AS AvgYearsSinceLastPromotion,
    AVG(YearsWithCurrManager) AS AvgYearsWithCurrManager,
    SUM(CASE WHEN Attrition = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS AttritionRate,
    AVG(DistanceFromHome_KM) AS AvgDistanceFromHome
FROM Employee;


select * from EmployeeSummaryStats;



                                  
     -- View for detailed Performance Rating summary statistics --

CREATE OR ALTER VIEW PerformanceRatingSummaryStats AS
SELECT 
    COUNT(*) AS TotalRatings,
    AVG(EnvironmentSatisfaction) AS AvgEnvironmentSatisfaction,
    STDEV(EnvironmentSatisfaction) AS StdDevEnvironmentSatisfaction,
    AVG(JobSatisfaction) AS AvgJobSatisfaction,
    STDEV(JobSatisfaction) AS StdDevJobSatisfaction,
    AVG(RelationshipSatisfaction) AS AvgRelationshipSatisfaction,
    STDEV(RelationshipSatisfaction) AS StdDevRelationshipSatisfaction,
    AVG(WorkLifeBalance) AS AvgWorkLifeBalance,
    STDEV(WorkLifeBalance) AS StdDevWorkLifeBalance,
    AVG(SelfRating) AS AvgSelfRating,
    STDEV(SelfRating) AS StdDevSelfRating,
    AVG(ManagerRating) AS AvgManagerRating,
    STDEV(ManagerRating) AS StdDevManagerRating,
    AVG(TrainingOpportunitiesWithinYear) AS AvgTrainingOpportunities,
    AVG(TrainingOpportunitiesTaken) AS AvgTrainingTaken,
    AVG(CAST(TrainingOpportunitiesTaken AS FLOAT) / NULLIF(TrainingOpportunitiesWithinYear, 0)) AS AvgTrainingUtilizationRate
FROM PerformanceRating;

select * from PerformanceRatingSummaryStats;



     -- View for Department-wise statistics --

CREATE OR ALTER VIEW DepartmentStats AS
SELECT 
    Department,
    COUNT(*) AS EmployeeCount,
    AVG(Salary) AS AvgSalary,
    STDEV(Salary) AS StdDevSalary,
    MIN(Salary) AS MinSalary,
    MAX(Salary) AS MaxSalary,
    AVG(YearsAtCompany) AS AvgYearsAtCompany,
    SUM(CASE WHEN Attrition = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS AttritionRate,
    AVG(Age) AS AvgAge,
    AVG(DistanceFromHome_KM) AS AvgDistanceFromHome
FROM Employee
GROUP BY Department;


select * from DepartmentStats;



     -- View for Job Role statistics --

CREATE OR ALTER VIEW JobRoleStats AS
SELECT 
    JobRole,
    COUNT(*) AS EmployeeCount,
    AVG(Salary) AS AvgSalary,
    STDEV(Salary) AS StdDevSalary,
    AVG(YearsAtCompany) AS AvgYearsAtCompany,
    AVG(YearsInMostRecentRole) AS AvgYearsInRole,
    SUM(CASE WHEN Attrition = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS AttritionRate
FROM Employee
GROUP BY JobRole;


select * from JobRoleStats;