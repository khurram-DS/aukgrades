
import streamlit as st
# Eda packages

import pandas as pd
import numpy as np

#Data viz packages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

#function

def main():
    
    title_container = st.container()
    col1, col2 ,  = st.columns([1,8])
    from PIL import Image
    image = Image.open('static/auk.jpg')
    with title_container:
        with col1:
            st.image(image, width=70)
        with col2:
            st.markdown('<h1 style="color: purple;">AUK Grades Distribution</h1>',
                           unsafe_allow_html=True)
        

    st.sidebar.image("static/confu.png", use_column_width=True)
    activites = ["About","Overall","Course","Section","Major","College"]
    choice =st.sidebar.selectbox("Select Activity",activites)
    def get_df(file):
      # get extension and read file
      extension = file.name.split('.')[1]
      if extension.upper() == 'CSV':
        df = pd.read_csv(file)
      elif extension.upper() == 'XLSX':
        df = pd.read_excel(file)
      
      return df
    file = st.file_uploader("Upload file", type=['csv' 
                                             ,'xlsx'])
    if not file:
        st.write("Upload a .csv or .xlsx file to get started")
        return
      
    df = get_df(file)
    
    df.rename(columns={'Total work': 'Total_work'},inplace=True)
   
    
    
    if choice == "About":
        st.subheader("About AUK ")
        st.text("""Welcome to the College of Business and Economics. 
The college prepares students through an American educational 
experience combining quality intellectual challenge and ethical 
professional practice for careers in Kuwait and beyond.""")
        
        st.subheader("About Auk Grading distribution")
        st.text("""A student will be awarded credit only once for any passed
course counted toward their degree or in the calculation of the GPA. Grades for 
all courses completed at AUK will be recorded on the student’s AUK transcript. 
Repeated courses will be designated to distinguish them from other courses.
The grade point average (GPA) is based on grades earned in courses 
at the American University of Kuwait, and is calculated based on the following 
equivalencies (the qualities of performance associated with the different grades 
are explained below Figure)""")
        
        from PIL import Image
        image = Image.open('static/grade.jpg')

        st.image(image, caption='AUK Grading Distribution')
        
        st.text('© American University of Kuwait')
#overall analysis    
    elif choice == "Overall":
        st.subheader("Overall Grades Analysis")
            
            
        
        if st.checkbox('Show Raw Data'):
            st.subheader('Raw Data')
            st.write(df)
            
        
            
        def determine_grade(Total_work):
                
            if Total_work >= 94 and Total_work <=100:
                return 'A'
            elif Total_work >= 90 and Total_work <=93.99999:
                return 'A-'
            elif Total_work >= 87 and Total_work <=89.999999:
                return 'B+'
            elif Total_work >= 84 and Total_work <=86.999999:
                return 'B'
            elif Total_work >= 80 and Total_work <=83.999999:
                return 'B-'
            elif Total_work >= 77 and Total_work <=79.999999:
                return 'C+'
            elif Total_work >= 74 and Total_work <=76.999999:
                return 'C'
            elif Total_work >= 70 and Total_work <=73.999999:
                return 'C-'
            elif Total_work >= 67 and Total_work <=69.999999:
                return 'D+'
            elif Total_work >= 64 and Total_work <=66.999999:
                return 'D'
            elif Total_work >= 60 and Total_work <=63.999999:
                return 'D-'
            elif Total_work >= 0 and Total_work <=59.999999:
                return 'F'
            else:
                return 'FN'
        df["Grade"] = df["Total_work"].apply(lambda x: determine_grade(x))
        
        if st.checkbox('Show Raw Data with Grades'):
            st.subheader('Raw Data With Grades')
            st.write(df)
            #downlad button
            st.text("Download the Above Data file by clicking on Download CSV")
            st.download_button(label='Download CSV',data=df.to_csv(),mime='text/csv')
            
        if st.checkbox('Grades Calculater'):
            st.subheader('Check your grades by entering your Total works')
            Total_work=st.number_input('Enter your Total work')
            st.write(Total_work)

            if Total_work >= 94 and Total_work <=100:
                st.write('✌ Your grade is [ A ] as your Total work is  {} '.format(Total_work)) 
            elif Total_work >= 90 and Total_work <=93.99999:
                st.write('✌ Your grade is [ A- ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 87 and Total_work <=89.999999:
                st.write('✔ Your grade is  [ B+ ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 84 and Total_work <=86.999999:
                st.write('✔ Your grade is [ B ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 80 and Total_work <=83.999999:
                st.write('✔ Your grade is [ B- ]  as your Total work is  {} '.format(Total_work))
            elif Total_work >= 77 and Total_work <=79.999999:
                st.write('✋ Your grade is [ C+ ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 74 and Total_work <=76.999999:
                st.write('✋ Your grade is [ C ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 70 and Total_work <=73.999999:
                st.write('✋ Your grade is [ C- ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 67 and Total_work <=69.999999:
                st.write('👎 Your grade is [ D+ ]  as your Total work is  {} '.format(Total_work))
            elif Total_work >= 64 and Total_work <=66.999999:
                st.write('👎 Your grade is [ D ] as your Total work is  {} '.format(Total_work))
            elif Total_work >= 60 and Total_work <=63.999999:
                st.write('👎 Your grade is [ D- ]  as your Total work is  {} '.format(Total_work))
            elif Total_work >= 0 and Total_work <=59.999999:
                st.write('✘ Your grade is [ F ] as your Total work is  {} '.format(Total_work))
            else:
                st.write('✘ Your grade is [ FN ] as your Total work is  {} '.format(Total_work))

            
        if st.checkbox('Show the Shape'):
            
            st.subheader("Shape of the Data = ")
            st.subheader("{} rows with {} columns".format(df.shape[0],df.shape[1]))
            
        
            
        if st.checkbox("Select Columns to show"):
            all_columns=df.columns.to_list()
            selected_columns= st.multiselect("Select Columns", all_columns)
            cnt=df[selected_columns]
            st.dataframe(cnt)
            st.text("Download the Above Data by clicking on Download CSV")
            st.download_button(label='Download CSV',data=cnt.to_csv(),mime='text/csv')
            
        if st.checkbox("Show summary statistics"):
            st.subheader('Summary statistics')
            st.write(df.describe().T)
            
        if st.checkbox("Statistics estimated on the input data"):
            st.text('Statistics estimated on the input data and computed using the')
            st.text('estimated parameters of the Normal distribution:')
            
            import numpy as np
            from scipy.stats import kurtosis, skew
            newdf=df._get_numeric_data()
            
            skew=skew(newdf)
            kur=kurtosis(newdf) \
            
            mean=newdf.mean().round(4)
            var=newdf.var().round(4)
            stat=pd.DataFrame({"mean":mean,
                  "variance":var,
                  "skewness":skew,
                  "kurtosis":kur})
            
            st.write(stat)
            
            
        if st.checkbox("Select Columns to see frequency table"):
            all_columns=df.columns.to_list()
            selected_columns= st.multiselect("Select Columns to see Counts and frequency", all_columns)
            cnt=df[selected_columns].value_counts()
            per=df[selected_columns].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            
            Course=pd.DataFrame({'counts': cnt,'Frequency %': per})
            Course.reset_index(inplace = True)
            Course.rename(columns={'index':'Course'},inplace=True)
            Course['Total_data']=Course['counts'].sum()
            st.dataframe(Course)
            
            st.text("Download the Above Data table by clicking on Download CSV")
            st.download_button(label='Download CSV',data=Course.to_csv(),mime='text/csv')
            
        if st.checkbox("Show Plots for Overall analysis"):
            #histgram plot
            st.subheader('Histogram Plot')
            import plotly.express as px
            fig2 = px.histogram(df, x="Total_work",
                   title='Histogram of Total_work',
                   labels={'Total_work':'Total_work'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['indianred'] # color of histogram bars
                   )
            
            st.write(fig2)
            
            
            
            # dist plot
            st.subheader('Distplot')
            import seaborn as sns

            import plotly.figure_factory as ff
            import numpy as np
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            from chart_studio import plotly as py
            import plotly
            from plotly import tools
            
            x = df['Total_work']
            hist_data = [x]
            group_labels = ['Distplot for Total Work'] # name of the dataset
            colors = ['rgb(20, 120, 4200)']
            fig3 = ff.create_distplot(hist_data, group_labels,bin_size=.6, colors=colors)
            fig3.update_layout(title_text='Distplot for Total_work')
            fig3.update_layout(
            autosize=False,
            width=850,
            height=550
            )
            fig3.update_xaxes(title_text='Total_work')
            fig3.update_yaxes(title_text='Density')
            
            st.write(fig3)
            
            # cdf plot
            st.subheader('CDF plot')
            import plotly.express as px
            fig4 = px.ecdf(df, x="Total_work",
                  title='CDF of Total_work',
                   labels={'Total_work':'Total_work'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['seagreen'] # color of histogram bars
                   )
            st.write(fig4)
            
            # plotting boxplot
            st.subheader('Box plot')
            st.text('Box Plot with Displaying The Underlying Data')
            import plotly.graph_objects as go

            fig5 = go.Figure(data=[go.Box(y=df['Total_work'],
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8, # relative position of points wrt box
            name='Total work',
            fillcolor='pink',
            
              )])
    
            fig5.update_layout(title_text='Boxplot for Total work')
            st.write(fig5)
            
            # plotting p-p plot
            st.subheader('Q-Q plot')
            st.text('plot for checking the distribution using Q-Q plot')
            st.text('We can also observe the distribution difference by clicking')
            from statsmodels.graphics.gofplots import qqplot

            qqplot_data = qqplot(df.Total_work, line='s').gca().lines
            fig = go.Figure()

            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })
            
            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }
            
            })
            
            
            fig['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 800,
                'height': 700,
            })
            
            st.write(fig)
  #countplot grades
            st.subheader('lets see some plot based on Grades')
            Grade = df.Grade
            counts = Grade.value_counts()
            percent = Grade.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            Grade=pd.DataFrame({'counts': counts,'Frequency %': percent})
            Grade.reset_index(inplace = True)
            Grade.rename(columns={'index':'Grade'},inplace=True)
            Grade['Total_data']=Grade['counts'].sum()
            import plotly.express as px
            fig10 = px.bar(Grade, x="Grade", y="counts", color="Frequency %", title="Frequency counts for Grades")
            st.write(fig10)
            st.subheader('Histogram of Grades distribution based on Total_work')
            trace0 = go.Histogram(
                    x=df.loc[df['Grade'] == 'A']['Total_work'], name='Grade A',
                    opacity=0.75
                )
            trace1 = go.Histogram(
                    x=df.loc[df['Grade'] == 'A-']['Total_work'], name='Grade A-',
                    opacity=0.75
                    
                )
            trace2 = go.Histogram(
                    x=df.loc[df['Grade'] == 'B+']['Total_work'], name='Grade B+',
                    opacity=0.65
                )
            trace3 = go.Histogram(
                    x=df.loc[df['Grade'] == 'B']['Total_work'], name='Grade B',
                    opacity=0.65
                )
            trace4 = go.Histogram(
                    x=df.loc[df['Grade'] == 'B-']['Total_work'], name='Grade B-',
                    opacity=0.65
                )
            trace5 = go.Histogram(
                    x=df.loc[df['Grade'] == 'C+']['Total_work'], name='Grade C+',
                    opacity=0.65
                )
            trace6 = go.Histogram(
                    x=df.loc[df['Grade'] == 'C']['Total_work'], name='Grade C-',
                    opacity=0.65
                )
            trace7 = go.Histogram(
                    x=df.loc[df['Grade'] == 'C-']['Total_work'], name='Grade C-',
                    opacity=0.65
                )
            trace8 = go.Histogram(
                    x=df.loc[df['Grade'] == 'D+']['Total_work'], name='Grade D+',
                    opacity=0.65
                )
            trace9 = go.Histogram(
                    x=df.loc[df['Grade'] == 'D']['Total_work'], name='Grade D',
                    opacity=0.65
                )
            trace10 = go.Histogram(
                    x=df.loc[df['Grade'] == 'D-']['Total_work'], name='Grade D-',
                    opacity=0.65
                )
            trace11 = go.Histogram(
                    x=df.loc[df['Grade'] == 'F']['Total_work'], name='Grade F',
                    opacity=0.65
                )
            trace12 = go.Histogram(
                    x=df.loc[df['Grade'] == 'FN']['Total_work'], name='Grade FN',
                    opacity=0.65
                )





            data = [trace0, trace1, trace2, trace3, trace4 ,trace5, trace6,trace7, trace8, trace9, trace10, trace11 ,trace12]

            layout = go.Layout(barmode='overlay', title='Histogram of Grades distribution based on Total_work')
            fig5 = go.Figure(data=data, layout=layout)
            st.write(fig5)
            
            
            
            
            # plotting box plot for grades based on total works
            st.subheader('Boxplot For Grades based on the Total_works')
            trace0 = go.Box(
                    y=df.loc[df['Grade'] == 'A']['Total_work'],
                    name = 'Grade (A)',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace1 = go.Box(
                    y=df.loc[df['Grade'] == 'A-']['Total_work'],
                    name = 'Grade (A-)',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            trace2 = go.Box(
                    y=df.loc[df['Grade'] == 'B+']['Total_work'],
                    name = 'Grade (B+)',
                    marker = dict(
                        color = 'rgb(12, 102, 14)',
                    )
                )
            trace3 = go.Box(
                    y=df.loc[df['Grade'] == 'B']['Total_work'],
                    name = 'Grade (B)',
                    marker = dict(
                        color = 'rgb(10, 0, 100)',
                    )
                )
            trace4 = go.Box(
                    y=df.loc[df['Grade'] == 'B-']['Total_work'],
                    name = 'Grade (B-)',
                    marker = dict(
                        color = 'rgb(100, 0, 10)',
                    )
                )
            trace5 = go.Box(
                    y=df.loc[df['Grade'] == 'C+']['Total_work'],
                    name = 'Grade (C+)',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace6 = go.Box(
                    y=df.loc[df['Grade'] == 'C']['Total_work'],
                    name = 'Grade (C)',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            trace7 = go.Box(
                    y=df.loc[df['Grade'] == 'C-']['Total_work'],
                    name = 'Grade (C-)',
                    marker = dict(
                        color = 'rgb(12, 102, 14)',
                    )
                )
            trace8 = go.Box(
                    y=df.loc[df['Grade'] == 'D+']['Total_work'],
                    name = 'Grade (D+)',
                    marker = dict(
                        color = 'rgb(10, 0, 100)',
                    )
                )
            trace9 = go.Box(
                    y=df.loc[df['Grade'] == 'D']['Total_work'],
                    name = 'Grade (D)',
                    marker = dict(
                        color = 'rgb(100, 0, 10)',
                    )
                )
            trace10 = go.Box(
                    y=df.loc[df['Grade'] == 'D-']['Total_work'],
                    name = 'Grade (D-)',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace11 = go.Box(
                    y=df.loc[df['Grade'] == 'F']['Total_work'],
                    name = 'Grade (F)',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            trace12 = go.Box(
                    y=df.loc[df['Grade'] == 'FN']['Total_work'],
                    name = 'Grade (FN)',
                    marker = dict(
                        color = 'rgb(12, 102, 14)',
                    )
                )
                
            data1 = [trace0, trace1, trace2, trace3, trace4 ,trace5, trace6,trace7, trace8, trace9, trace10, trace11 ,trace12]
            layout1 = go.Layout(
                    title = "Boxplot of Grades by Total_work"
                )
                
            fig6 = go.Figure(data1,layout1)
                
            st.write(fig6)
            
            
            st.subheader(" That's all for the Overall analysis, Hope you got lots of details from it ✌")
            st.subheader("Select the different activity to Explore the analysis 👉 ")
          
            st.text('© American University of Kuwait')
            
        
        
            #based on course analysis    
        
        
    elif choice == "Course":
        st.subheader("Course Based Grades Analysis")
        
        if st.checkbox("Show Course frequency table"):
            Course = df.Course
            counts = Course.value_counts()
            percent = Course.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            Course=pd.DataFrame({'counts': counts,'Frequency %': percent})
            Course.reset_index(inplace = True)
            Course.rename(columns={'index':'Course'},inplace=True)
            Course['Total_data']=Course['counts'].sum()
            st.dataframe(Course)
            
            st.text("Download the Above Data table by clicking on Download CSV")
            st.download_button(label='Download CSV',data=Course.to_csv(),mime='text/csv')
            import plotly.express as px
            fig = px.bar(Course, x="Course", y="counts", color="Frequency %", title="Frequency counts for different Courses")
            st.write(fig)
            
        if st.checkbox("Show Summary statistics"):
            stat=df.groupby('Course')['Total_work'].describe()
            st.subheader('Summary statistics for Course based on the Total_work')
            st.write(stat)
        
        if st.checkbox("Show t-test for two independent samples / Two-tailed test for the course"):
            
            from pylab import rcParams
            from scipy.stats import f_oneway
            from scipy.stats import ttest_ind
            import seaborn as sns
            import numpy as np
            import warnings            
            warnings.filterwarnings("ignore")
            course_a=df.loc[df['Course'] == 'Course A']['Total_work']
            course_b=df.loc[df['Course'] == 'Course B']['Total_work']
            course_c=df.loc[df['Course'] == 'Course C']['Total_work']
            rcParams['figure.figsize'] = 20,10
            rcParams['font.size'] = 30
            sns.set()
            np.random.seed(8)
            
            st.subheader('T-test result table')
            st.text('➊ For course_a and course_b')
            import pingouin as pg

            st.write(pg.ttest(course_a, course_b, correction=False))
            
            st.text('➋ For course_a and course_c')
            import pingouin as pg

            st.write( pg.ttest(course_a, course_c, correction=False))
            st.text('➌ For course_b and course_c')
            import pingouin as pg

            st.write(pg.ttest(course_b, course_c, correction=False))
            
            
                
# ttest course a and course b
            st.subheader('Lets understand the t-test result for Course more deeply')   
            if st.checkbox("Show the t-test for Total work | Course-Course A - Total work | Course-Course B"):
                st.subheader('Lets check the length of Course_a and course_b')
                st.text("Length of Course_a")
                st.write(len(course_a))
                
                st.text("Length of Course_b")
                st.write(len(course_b))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Course-Course A - Total work | Course-Course B):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Course_a using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # course_b               
                st.text(' Checking normality of data for Course_b using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Course-Course A - Total work | Course-Course B"): ' )
                import pingouin as pg

                res = pg.ttest(course_a, course_b, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Course-Course A - Total work | Course-Course B):')
                import matplotlib.pyplot as plt
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(course_a)
                ax2 = sns.distplot(course_b)
                plt.axvline(np.mean(course_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(course_b), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
                
#t test for course a and course c
                
                # ttest course a and Course C
                
            if st.checkbox("Show the t-test for Total work | Course-Course A - Total work | Course-Course C"):
                st.subheader('Lets check the length of Course_a and course_c')
                st.text("Length of Course_a")
                st.write(len(course_a))
                
                st.text("Length of course_c")
                st.write(len(course_c))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Course-Course A - Total work | Course-Course C):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Course_a using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # course_c               
                st.text(' Checking normality of data for course_c using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_c)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_c), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Course-Course A - Total work | Course-Course C"): ' )
                import pingouin as pg

                res = pg.ttest(course_a, course_c, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Course-Course A - Total work | Course-Course C):')
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(course_a)
                ax2 = sns.distplot(course_c)
                plt.axvline(np.mean(course_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(course_c), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
            
#t test for course c and course b
                
                
            if st.checkbox("Show the t-test for Total work | Course-Course B - Total work | Course-Course C"):
                st.subheader('Lets check the length of course_b and course_c')
                st.text("Length of course_b")
                st.write(len(course_b))
                
                st.text("Length of course_c")
                st.write(len(course_c))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Course-Course B - Total work | Course-Course C):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for course_b using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # course_c               
                st.text(' Checking normality of data for course_c using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(course_c)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(course_c), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Course-Course B - Total work | Course-Course C"): ' )
                import pingouin as pg

                res = pg.ttest(course_b, course_c, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Course-Course B - Total work | Course-Course C):')
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(course_b)
                ax2 = sns.distplot(course_c)
                plt.axvline(np.mean(course_b), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(course_c), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
    #plot
        if st.checkbox("Show Plots for different course, based on total work"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            import chart_studio.plotly as py
            import plotly.offline as py
            import plotly
            from plotly import tools
            trace0 = go.Box(
                    y=df.loc[df['Course'] == 'Course A']['Total_work'],
                    name = 'Course A',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace1 = go.Box(
                    y=df.loc[df['Course'] == 'Course B']['Total_work'],
                    name = 'Course B',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            trace2 = go.Box(
                    y=df.loc[df['Course'] == 'Course C']['Total_work'],
                    name = 'Course C',
                    marker = dict(
                        color = 'rgb(12, 102, 14)',
                    )
                )
                
            data1 = [trace0, trace1, trace2]
            layout1 = go.Layout(
                    title = "Boxplot of Courses based on Total_work"
                )
                
            fig1 = go.Figure(data1,layout1)
            st.write(fig1)
    #histogram plot        
            st.subheader('Histogram plot for Total_work for all the courses')
            
            trace0 = go.Histogram(
                    x=df.loc[df['Course'] == 'Course A']['Total_work'], name='With Course A',
                    opacity=0.75
                )
            trace1 = go.Histogram(
                    x=df.loc[df['Course'] == 'Course B']['Total_work'], name='with Course B',
                    opacity=0.75
                    
                )
            trace2 = go.Histogram(
                    x=df.loc[df['Course'] == 'Course C']['Total_work'], name='with Course C',
                    opacity=0.65
                )
            data = [trace0, trace1,trace2]
                
            layout = go.Layout(barmode='overlay', title='Histogram of Total_work for all the course')
            fig2 = go.Figure(data=data, layout=layout)
            st.write(fig2)
            
            st.subheader("Thats all for course based analysis")
            st.subheader("Navigate the activity to see other analysis 👉")
       
            st.text('© American University of Kuwait')
        
        
        
     # section based analysis   
        
        
        
        
        
        
        
        
    elif choice == "Section":
        st.subheader("Section Based Grades Analysis")
        if st.checkbox("Show Section frequency table"):
            Section = df.Section
            counts = Section.value_counts()
            percent = Section.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            Section=pd.DataFrame({'counts': counts,'Frequency %': percent})
            Section.reset_index(inplace = True)
            Section.rename(columns={'index':'Section'},inplace=True)
            Section['Total_data']=Section['counts'].sum()
            st.dataframe(Section)
            
            st.text("Download the Above Data table by clicking on Download CSV")
            st.download_button(label='Download CSV',data=Section.to_csv(),mime='text/csv')
            import plotly.express as px
            fig1 = px.bar(Section, x="Section", y="counts", color="Frequency %", title="Frequency counts for different Sections")
            st.write(fig1)
            
        if st.checkbox("Show Summary statistics"):
            stat=df.groupby('Section')['Total_work'].describe()
            st.subheader('Summary statistics for Section based on the Total_work')
            st.write(stat)
        
        if st.checkbox("Show t-test for two independent samples / Two-tailed test for the Section"):
            
            from pylab import rcParams
            from scipy.stats import f_oneway
            from scipy.stats import ttest_ind
            import seaborn as sns
            import numpy as np
            import warnings            
            warnings.filterwarnings("ignore")
            Section_a=df.loc[df['Section'] == 'Section 01']['Total_work']
            Section_b=df.loc[df['Section'] == 'Section 02']['Total_work']            
            rcParams['figure.figsize'] = 20,10
            rcParams['font.size'] = 30
            sns.set()
            np.random.seed(8)
            
            st.subheader('T-test result table')
            st.text('➊ For Section 01 and Section 02')
            import pingouin as pg

            st.write(pg.ttest(Section_a, Section_b, correction=False))
        
        # ttest Section a and Section b
            st.subheader('Lets understand the t-test result for Section more deeply') 
            
            if st.checkbox("Show the t-test for Total work | Section-Section 01 - Total work | Section-Section 02"):
                st.subheader('Lets check the length of Section 01 and Section 02')
                st.text("Length of Section 01")
                st.write(len(Section_a))
                
                st.text("Length of Section 02")
                st.write(len(Section_b))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Section-Section 01 - Total work | Section-Section 02):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Section 01 using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Section_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Section_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # Section_b               
                st.text(' Checking normality of data for Section 02 using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Section_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Section_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Section-Section 01 - Total work | Section-Section 02"): ' )
                import pingouin as pg

                res = pg.ttest(Section_a, Section_b, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Section-Section 01 - Total work | Section-Section 02):')
                import matplotlib.pyplot as plt
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(Section_a)
                ax2 = sns.distplot(Section_b)
                plt.axvline(np.mean(Section_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(Section_b), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
        #boxplot
        if st.checkbox("Show Plots for different Section, based on total work"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            import chart_studio.plotly as py
            import plotly.offline as py
            import plotly
            from plotly import tools
            trace0 = go.Box(
                    y=df.loc[df['Section'] == 'Section 01']['Total_work'],
                    name = 'Section 01',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace1 = go.Box(
                    y=df.loc[df['Section'] == 'Section 02']['Total_work'],
                    name = 'Section 02',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            
                
            data1 = [trace0, trace1]
            layout1 = go.Layout(
                    title = "Boxplot of Section in terms of Total_work"
                )
                
            fig1 = go.Figure(data1,layout1)
            st.write(fig1)
            
            #histogram plot        
            st.subheader('Histogram plot for Total_work for all the sections')
            
            trace0 = go.Histogram(
                    x=df.loc[df['Section'] == 'Section 01']['Total_work'], name='With Section 01',
                    opacity=0.75
                )
            trace1 = go.Histogram(
                    x=df.loc[df['Section'] == 'Section 02']['Total_work'], name='with Section 02',
                    opacity=0.75
                    
                )
           
            data = [trace0, trace1]
                
            layout = go.Layout(barmode='overlay', title='Histogram of Total_work for all the Section')
            fig2 = go.Figure(data=data, layout=layout)
            st.write(fig2)
            
            st.subheader("Thats all for Section based analysis")
            st.subheader("Navigate the activity to see other analysis 👉")
            
            st.text('© American University of Kuwait')
        
        
        
        
        
        
    # major analysis  
          
        
    elif choice == "Major":
        st.subheader("Major's Subject Based Grades Analysis")
        
        if st.checkbox("Show Major frequency table"):
            Major = df.Major
            counts = Major.value_counts()
            percent = Major.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            Major=pd.DataFrame({'counts': counts,'Frequency %': percent})
            Major.reset_index(inplace = True)
            Major.rename(columns={'index':'Major'},inplace=True)
            Major['Total_data']=Major['counts'].sum()
            st.dataframe(Major)
            
            st.text("Download the Above Data table by clicking on Download CSV")
            st.download_button(label='Download CSV',data=Major.to_csv(),mime='text/csv')
            import plotly.express as px
            fig = px.bar(Major, x="Major", y="counts", color="Frequency %", title="Frequency counts for different Majors")
            st.write(fig)
            
        if st.checkbox("Show Summary statistics"):
            stat=df.groupby('Major')['Total_work'].describe()
            st.subheader('Summary statistics for Marketingased on the Total_work')
            st.write(stat)
        
        if st.checkbox("Show t-test for two independent samples / Two-tailed test for the Major"):
            
            from pylab import rcParams
            from scipy.stats import f_oneway
            from scipy.stats import ttest_ind
            import seaborn as sns
            import numpy as np
            import warnings            
            warnings.filterwarnings("ignore")
            Major_a=df.loc[df['Major'] == 'Engeneering']['Total_work']
            Major_b=df.loc[df['Major'] == 'Marketing']['Total_work']
            Major_c=df.loc[df['Major'] == 'Finance']['Total_work']
            rcParams['figure.figsize'] = 20,10
            rcParams['font.size'] = 30
            sns.set()
            np.random.seed(8)
            
            st.subheader('T-test result table')
            st.text('➊ For Major_a and Major_b')
            import pingouin as pg

            st.write(pg.ttest(Major_a, Major_b, correction=False))
            
            st.text('➋ For Major_a and Major_c')
            import pingouin as pg

            st.write( pg.ttest(Major_a, Major_c, correction=False))
            st.text('➌ For Major_b and Major_c')
            import pingouin as pg

            st.write(pg.ttest(Major_b, Major_c, correction=False))
            
            
                
# ttest Engeneering and Marketing
            st.subheader('Lets understand the t-test result for Major more deeply')   
            if st.checkbox("Show the t-test for Total work | Major-Engeneering - Total work | Major-Marketing"):
                st.subheader('Lets check the length of Major_a and Major_b')
                st.text("Length of Major_a")
                st.write(len(Major_a))
                
                st.text("Length of Major_b")
                st.write(len(Major_b))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Major-Engeneering - Total work | Major-Marketing):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Major_a using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # Major_b               
                st.text(' Checking normality of data for Major_b using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Major-Engeneering - Total work | Major-Marketing"): ' )
                import pingouin as pg

                res = pg.ttest(Major_a, Major_b, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Major-Engeneering - Total work | Major-Marketing):')
                import matplotlib.pyplot as plt
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(Major_a)
                ax2 = sns.distplot(Major_b)
                plt.axvline(np.mean(Major_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(Major_b), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
                
#t test for Engeneering and Finance
                
                # ttest Engeneering and Finance
                
            if st.checkbox("Show the t-test for Total work | Major-Engeneering - Total work | Major-Finance"):
                st.subheader('Lets check the length of Major_a and Major_c')
                st.text("Length of Major_a")
                st.write(len(Major_a))
                
                st.text("Length of Major_c")
                st.write(len(Major_c))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Major-Engeneering - Total work | Major-Finance):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Major_a using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # Major_c               
                st.text(' Checking normality of data for Major_c using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_c)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_c), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Major-Engeneering - Total work | Major-Finance"): ' )
                import pingouin as pg

                res = pg.ttest(Major_a, Major_c, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Major-Engeneering - Total work | Major-Finance):')
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(Major_a)
                ax2 = sns.distplot(Major_c)
                plt.axvline(np.mean(Major_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(Major_c), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
            
#t test for Finance and Marketing
                
                
            if st.checkbox("Show the t-test for Total work | Major-Marketing - Total work | Major-Finance"):
                st.subheader('Lets check the length of Major_b and Major_c')
                st.text("Length of Major_b")
                st.write(len(Major_b))
                
                st.text("Length of Major_c")
                st.write(len(Major_c))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | Major-Marketing - Total work | Major-Finance):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Major_b using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # Major_c               
                st.text(' Checking normality of data for Major_c using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(Major_c)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(Major_c), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | Major-Marketing - Total work | Major-Finance"): ' )
                import pingouin as pg

                res = pg.ttest(Major_b, Major_c, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | Major-Marketing - Total work | Major-Finance):')
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(Major_b)
                ax2 = sns.distplot(Major_c)
                plt.axvline(np.mean(Major_b), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(Major_c), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
    #plot
        if st.checkbox("Show Plots for different Major, based on total work"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            import chart_studio.plotly as py
            import plotly.offline as py
            import plotly
            from plotly import tools
            trace0 = go.Box(
                    y=df.loc[df['Major'] == 'Engeneering']['Total_work'],
                    name = 'Engeneering',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace1 = go.Box(
                    y=df.loc[df['Major'] == 'Marketing']['Total_work'],
                    name = 'Marketing',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            trace2 = go.Box(
                    y=df.loc[df['Major'] == 'Finance']['Total_work'],
                    name = 'Finance',
                    marker = dict(
                        color = 'rgb(12, 102, 14)',
                    )
                )
                
            data1 = [trace0, trace1, trace2]
            layout1 = go.Layout(
                    title = "Boxplot of Majors based on Total_work"
                )
                
            fig1 = go.Figure(data1,layout1)
            st.write(fig1)
    #histogram plot        
            st.subheader('Histogram plot for Total_work for all the Majors')
            
            trace0 = go.Histogram(
                    x=df.loc[df['Major'] == 'Engeneering']['Total_work'], name='With Engeneering',
                    opacity=0.75
                )
            trace1 = go.Histogram(
                    x=df.loc[df['Major'] == 'Marketing']['Total_work'], name='with Marketing',
                    opacity=0.75
                    
                )
            trace2 = go.Histogram(
                    x=df.loc[df['Major'] == 'Finance']['Total_work'], name='with Finance',
                    opacity=0.65
                )
            data = [trace0, trace1,trace2]
                
            layout = go.Layout(barmode='overlay', title='Histogram of Total_work for all the Major')
            fig2 = go.Figure(data=data, layout=layout)
            st.write(fig2)
            
            st.subheader("Thats all for Major based analysis")
            st.subheader("Navigate the activity to see other analysis 👉")
            
            st.text('© American University of Kuwait')
        
        
        
 #college based analysis       
        
            
        
    elif choice == "College":
        st.subheader("College Based Grades Analysis")
        
        if st.checkbox("Show College frequency table"):
            College = df.College
            counts = College.value_counts()
            percent = College.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            College=pd.DataFrame({'counts': counts,'Frequency %': percent})
            College.reset_index(inplace = True)
            College.rename(columns={'index':'College'},inplace=True)
            College['Total_data']=College['counts'].sum()
            st.dataframe(College)
            
            st.text("Download the Above Data table by clicking on Download CSV")
            st.download_button(label='Download CSV',data=College.to_csv(),mime='text/csv')
            import plotly.express as px
            fig1 = px.bar(College, x="College", y="counts", color="Frequency %", title="Frequency counts for different Colleges")
            st.write(fig1)
            
        if st.checkbox("Show Summary statistics"):
            stat=df.groupby('College')['Total_work'].describe()
            st.subheader('Summary statistics for College based on the Total_work')
            st.write(stat)
        
        if st.checkbox("Show t-test for two independent samples / Two-tailed test for the College"):
            
            from pylab import rcParams
            from scipy.stats import f_oneway
            from scipy.stats import ttest_ind
            import seaborn as sns
            import numpy as np
            import warnings            
            warnings.filterwarnings("ignore")
            College_a=df.loc[df['College'] == 'Eng.']['Total_work']
            College_b=df.loc[df['College'] == 'Business']['Total_work']            
            rcParams['figure.figsize'] = 20,10
            rcParams['font.size'] = 30
            sns.set()
            np.random.seed(8)
            
            st.subheader('T-test result table')
            st.text('➊ For Eng. and Business')
            import pingouin as pg

            st.write(pg.ttest(College_a, College_b, correction=False))
        
        # ttest College a and College b
            st.subheader('Lets understand the t-test result for College more deeply') 
            
            if st.checkbox("Show the t-test for Total work | College-Eng. - Total work | College-Business"):
                st.subheader('Lets check the length of Eng. and Business')
                st.text("Length of Eng.")
                st.write(len(College_a))
                
                st.text("Length of Business")
                st.write(len(College_b))
                st.subheader("t-test for two independent samples / Two-tailed test (Total work | College-Eng. - Total work | College-Business):")
                st.subheader('Making some Asuumations')
                st.text('Assumption 1: Are the two samples independent?')
                st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                
                
                st.subheader('Checking the Normality of Data')
                st.text(' Checking normality of data for Eng. using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(College_a)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(College_a), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
 # College_b               
                st.text(' Checking normality of data for Business using shapiro test')
                
                from scipy.stats import shapiro
                stat, p = shapiro(College_b)
                
                # interpret
                alpha = 0.05
                if p > alpha:
                    msg = 'Sample looks Gaussian (fail to reject H0)'
                else:
                    msg = 'Sample does not look Gaussian (reject H0)'
                
                result_mat = [
                    ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                    [len(College_b), stat, p, msg]
                ]
                import plotly.figure_factory as ff
                swt_table = ff.create_table(result_mat)
                swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                swt_table['layout']['height']=200
                swt_table['layout']['margin']['t']=50
                swt_table['layout']['margin']['b']=50
                
                #py.iplot(swt_table, filename='shapiro-wilk-table')
                st.write(swt_table)
                
            #t-test
                st.subheader('lets see the t-test results for Total work | College-Eng. - Total work | College-Business"): ' )
                import pingouin as pg

                res = pg.ttest(College_a, College_b, correction=False)
                st.write(res)
                
                st.download_button(label='Download t-test results',data=res.to_csv(),mime='text/csv')
        #test interpretation
                st.subheader('Test interpretation: results ')
                st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                st.text('● T is simply the calculated difference represented in units of standard error')
                st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                
                
                alpha=0.05
                if res['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.subheader('Plot for Two-tailed test (Total work | College-Eng. - Total work | College-Business):')
                import matplotlib.pyplot as plt
                def plot_distribution(inp):
                    plt.figure()
                    ax = sns.distplot(inp)
                    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                    _, max_ = plt.ylim()
                    plt.text(
                        inp.mean() + inp.mean() / 10,
                        max_ - max_ / 10,
                        "Mean: {:.2f}".format(inp.mean()),
                    )
                    return plt.figure
                
                ax1 = sns.distplot(College_a)
                ax2 = sns.distplot(College_b)
                plt.axvline(np.mean(College_a), color='b', linestyle='dashed', linewidth=5)
                plt.axvline(np.mean(College_b), color='orange', linestyle='dashed', linewidth=5)
                st.pyplot(plt)
                showPyplotGlobalUse = False
        #boxplot
        if st.checkbox("Show Plots for different College, based on total work"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            import chart_studio.plotly as py
            import plotly.offline as py
            import plotly
            from plotly import tools
            trace0 = go.Box(
                    y=df.loc[df['College'] == 'Eng.']['Total_work'],
                    name = 'Eng.',
                    marker = dict(
                        color = 'rgb(214, 12, 140)',
                    )
                )
            trace1 = go.Box(
                    y=df.loc[df['College'] == 'Business']['Total_work'],
                    name = 'Business',
                    marker = dict(
                        color = 'rgb(0, 128, 128)',
                    )
                )
            
                
            data1 = [trace0, trace1]
            layout1 = go.Layout(
                    title = "Boxplot of College in terms of Total_work"
                )
                
            fig1 = go.Figure(data1,layout1)
            st.write(fig1)
            
            #histogram plot        
            st.subheader('Histogram plot for Total_work for all the Colleges')
            
            trace0 = go.Histogram(
                    x=df.loc[df['College'] == 'Eng.']['Total_work'], name='With Eng.',
                    opacity=0.75
                )
            trace1 = go.Histogram(
                    x=df.loc[df['College'] == 'Business']['Total_work'], name='with Business',
                    opacity=0.75
                    
                )
           
            data = [trace0, trace1]
                
            layout = go.Layout(barmode='overlay', title='Histogram of Total_work for all the College')
            fig2 = go.Figure(data=data, layout=layout)
            st.write(fig2)
            
           
            st.subheader(" Thats all for the AUK Grades Analysis ")
            
            st.text('This Dashboard has been implemented using streamlit & python packages. ')
        
        
            st.text('© American University of Kuwait')
        
        
      

if __name__=='__main__':
    main()
        
        