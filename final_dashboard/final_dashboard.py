import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate

# Load the data
df = pd.read_excel("C:/OneDrive/Desktop/final_dashboard/Dataset.xlsx")
print(df.head())
print(df.columns)

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('HR Analysis Dashboard'),
    
    html.Div([
        dcc.Dropdown(
            id='department-dropdown',
            options=[{'label': i, 'value': i} for i in df['Department'].unique()],
            value='All',
            multi=False,
            placeholder='Select a department'
        ),
        dcc.Dropdown(
            id='job-role-dropdown',
            options=[{'label': i, 'value': i} for i in df['JobRole'].unique()],
            value='All',
            multi=False,
            placeholder='Select a job role'
        )
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        dcc.Graph(id='salary-histogram', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='age-salary-scatter', style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='department-pie', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='education-bar', style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='job-satisfaction-heatmap', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='years-company-box', style={'width': '50%', 'display': 'inline-block'})
    ]),  
    
    html.Div([
        dcc.Graph(id='attrition-pie', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='performance-rating-bar', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='work-life-balance-bar', style={'width': '33%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='salary-performance-scatter', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='years-promotion-box', style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='attrition-factors-heatmap', style={'width': '100%', 'display': 'inline-block'})
    ])          
])


# Define callback for updating graphs

@app.callback(
    [Output('salary-histogram', 'figure'),
    Output('age-salary-scatter', 'figure'),
    Output('department-pie', 'figure'),
    Output('education-bar', 'figure'),
    Output('job-satisfaction-heatmap', 'figure'),
    Output('years-company-box', 'figure'),
    Output('attrition-pie', 'figure'),
    Output('performance-rating-bar', 'figure'),
    Output('work-life-balance-bar', 'figure'),
    Output('salary-performance-scatter', 'figure'),
    Output('years-promotion-box', 'figure'),
    Output('attrition-factors-heatmap', 'figure')],
    [Input('department-dropdown', 'value'),
    Input('job-role-dropdown', 'value')]
)


def update_graphs(selected_department, selected_job_role):
    print(f"Callback triggered with department: {selected_department}, job role: {selected_job_role}")
    
    if selected_department == 'All' and selected_job_role == 'All':
        filtered_df = df
    elif selected_department != 'All' and selected_job_role == 'All':
        filtered_df = df[df['Department'] == selected_department]
    elif selected_department == 'All' and selected_job_role != 'All':
        filtered_df = df[df['JobRole'] == selected_job_role]
    else:
        filtered_df = df[(df['Department'] == selected_department) & (df['JobRole'] == selected_job_role)]
    
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    
    if filtered_df.empty:
        print("Filtered dataframe is empty")
        raise PreventUpdate

    # Salary Histogram
    salary_hist = px.histogram(filtered_df, x='Salary', title='Salary Distribution')
    
    # Age vs Salary Scatter Plot
    age_salary_scatter = px.scatter(filtered_df, x='Age', y='Salary', 
                                    color='Gender', title='Age vs Salary')
    
    # Department Distribution Pie Chart
    dept_counts = filtered_df['Department'].value_counts()
    department_pie = px.pie(values=dept_counts.values, names=dept_counts.index, 
                            title='Employee Distribution by Department')
    
    # Education Level Bar Chart
    edu_counts = filtered_df['EducationLevel'].value_counts()
    education_bar = px.bar(x=edu_counts.index, y=edu_counts.values, 
                        title='Education Level Distribution')
    
    # Job Satisfaction Heatmap
    job_satisfaction = filtered_df.pivot_table(values='JobSatisfaction', 
                                            index='Department', 
                                            columns='JobRole', 
                                            aggfunc='mean')
    job_satisfaction_heatmap = px.imshow(job_satisfaction, 
                                        title='Job Satisfaction by Department and Role')
    
    # Years at Company Box Plot
    years_company_box = px.box(filtered_df, x='Department', y='YearsAtCompany', 
                            title='Years at Company by Department')
    
    # Attrition Pie Chart
    attrition_counts = filtered_df['Attrition'].value_counts()
    attrition_pie = px.pie(values=attrition_counts.values, names=attrition_counts.index, 
                        title='Employee Attrition', color_discrete_sequence=['#66b3ff', '#ff9999'])

    # Performance Rating Bar Chart
    perf_rating_counts = filtered_df['ManagerRatingLevel'].value_counts().sort_index()
    performance_rating_bar = px.bar(x=perf_rating_counts.index, y=perf_rating_counts.values, 
                                    title='Performance Ratings Distribution',
                                    labels={'x': 'Rating', 'y': 'Count'},
                                    color_discrete_sequence=['#99ff99'])

    # Work-Life Balance Bar Chart
    work_life_balance_counts = filtered_df['WorkLifeBalance'].value_counts().sort_index()
    work_life_balance_bar = px.bar(x=work_life_balance_counts.index, y=work_life_balance_counts.values, 
                                title='Work-Life Balance Distribution',
                                labels={'x': 'Work-Life Balance Score', 'y': 'Count'},
                                color_discrete_sequence=['#ffcc99'])

    # Salary vs Performance Scatter Plot
    salary_performance_scatter = px.scatter(filtered_df, x='Salary', y='ManagerRating', 
                                            color='Department', size='YearsAtCompany',
                                            title='Salary vs Performance Rating',
                                            labels={'Salary': 'Salary ($)', 'ManagerRating': 'Manager Rating'})

    # Years Since Last Promotion Box Plot
    years_promotion_box = px.box(filtered_df, x='Department', y='YearsSinceLastPromotion', 
                                title='Years Since Last Promotion by Department',
                                color='Department')

    # Attrition Factors Heatmap
    attrition_factors = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 
                        'RelationshipSatisfaction', 'ManagerRating']
    attrition_corr = filtered_df[attrition_factors].corr()
    attrition_factors_heatmap = px.imshow(attrition_corr, title='Correlation of Attrition Factors',
                                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    print("Graphs created successfully")
    return (salary_hist, age_salary_scatter, department_pie, 
            education_bar, job_satisfaction_heatmap, years_company_box, 
            attrition_pie, performance_rating_bar, work_life_balance_bar, 
            salary_performance_scatter, years_promotion_box, attrition_factors_heatmap)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)