##############################################################################
# Understanding the impact of Growth and Margin profile on B2B SaaS Valuations
# Dataset: 106 B2B SaaS companies
#   Author: Ramu Arunachalam (ramu@acapital.com)
#   Created: 06/20/21
#   Datset last updated: 06/09/21
###############################################################################

import joblib as jl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_date = '2021-06-11'
saas_filename_all = f'{file_date}-comps_B2B_ALL.csv'
saas_filename_high_growth = f'{file_date}-comps_B2B_High_Growth.csv'


def get_scatter_fig(df, x, y):
    fig = px.scatter(df,
                     x=x,
                     y=y,
                     hover_data=['Name'],
                     title=f'{y} vs {x}')
    df_r = df[[y] + [x]].dropna()
    model = sm.OLS(df_r[y], sm.add_constant(df_r[x])).fit()
    regline = sm.OLS(df_r[y], sm.add_constant(df_r[x])).fit().fittedvalues
    fig.add_trace(go.Scatter(x=df_r[x],
                             y=model.predict(),
                             mode='lines',
                             marker_color='black',
                             name='Best-fit',
                             line=dict(width=4, dash='dot')))
    return fig


latex_dict = {'EV / NTM Revenue': r'''\frac{EV}{Rev_{NTM}}''',
              'EV / 2021 Revenue': r'''\frac{EV}{Rev_{2021}}''',
              'EV / NTM Gross Profit': r'''\frac{EV}{GP_{NTM}}''',
              'EV / 2021 Gross Profit': r'''\frac{EV}{GP_{2021}}''',
              'NTM Revenue Growth': r'''Rev\,Growth_{NTM}''',
              '2021 Revenue Growth': r'''Rev\,Growth_{2021}''',
              'Growth adjusted EV / LTM Revenue': r'''\frac{EV}{Rev_{LTM}}\cdot\frac{1}{Growth_{NTM}}''',
              'Growth adjusted EV / 2020 Revenue': r'''\frac{EV}{Rev_{2020}}\cdot\frac{1}{Growth_{2021}}'''
              }


class RegressionInput:
    def __init__(self, df, x_vars, y_var):
        self.df = df
        self.x_vars = x_vars
        self.y_var = y_var
        self._hash = tuple([jl.hash(df), tuple(self.x_vars), tuple([self.y_var])])
        return

    def hash(self):
        return self._hash


class RegressionOutput:
    def __init__(self, reg_input, df_r, model, df_pvalues, vif_data):
        self.df_r = df_r
        self.model = model
        self.df_pvalues = df_pvalues
        self.vif_data = vif_data
        self.plot_figs = dict()

        # Regression equation
        self.eq_str = latex_dict.get(reg_input.y_var, reg_input.y_var) + r'''= \beta_0'''

        for x, i in zip(reg_input.x_vars, range(len(reg_input.x_vars))):
            self.eq_str += rf'''+\beta_{{{i + 1}}}\cdot {{{latex_dict.get(x, x)}}}'''
        self.eq_str += r'''+\epsilon'''
        self.eq_str = self.eq_str.replace("%", "\%").replace("&", "\&").replace("$", "\$")

        # Compute regression plots and save them
        # Plot residuals
        for x in reg_input.x_vars:
            self.plot_figs[x] = sm.graphics.plot_regress_exog(self.model, x)

        self._hash = tuple([reg_input.hash(), jl.hash(df_r), id(model), jl.hash(df_pvalues), jl.hash(vif_data)])
        return

    def hash(self):
        return self._hash


def reg_input_hash(reg_input):
    h = reg_input.hash()
    # st.info(f"reg_input_hash: h = {h}")
    return h


def reg_output_hash(reg_output):
    h = reg_output.hash()
    # st.info(f'reg_output hash = {h}')
    return h


class Experiment:
    id_num = 0

    def __init__(self):
        df_main = pd.read_csv(saas_filename_all)

        # Clean: 2x --> 2, 80% --> 80, $3,000 --> 3000
        df_obj = df_main[set(df_main.columns) - {'Name'}].select_dtypes(['object'])
        df_main[df_obj.columns] = df_obj \
            .apply(lambda x: x.str.strip('x')) \
            .apply(lambda x: x.str.strip('%')) \
            .replace(',', '', regex=True) \
            .replace('\$', '', regex=True)

        cols = df_main.columns
        for c in cols:
            try:
                df_main[c] = pd.to_numeric(df_main[c])
            except:
                pass

        df_main['2021 Revenue Growth'] = (df_main['2021 Analyst Revenue Estimates'].astype(float) / df_main[
            '2020 Revenue'].astype(
            float) - 1) * 100

        self.tickers_all = list(df_main[df_main['Name'].isin(['Median', 'Mean']) == False]['Name'])
        df_main_hg = pd.read_csv(saas_filename_high_growth)
        self.tickers_hg = list(df_main_hg[df_main_hg['Name'].isin(['Median', 'Mean']) == False]['Name'])
        self.tickers_excl_hg = list(set(self.tickers_all) - set(self.tickers_hg))

        self.df_main = df_main
        self.df = df_main
        self.reg_input = None
        self.reg_output = None

        return

    def filter(self, by):
        if by == 'High growth only':
            tickers = self.tickers_hg
        elif by == 'All (excl. high growth)':
            tickers = self.tickers_excl_hg
        else:
            tickers = self.tickers_all

        self.df = self.df_main[self.df_main['Name'].isin(tickers)]  # type of dataset
        return self

    def set_fwd_timeline(self, type):
        self.rev_g = f'{type} Revenue Growth'
        self.rev_mult = f'EV / {type} Revenue'
        self.gp_mult = f'EV / {type} Gross Profit'
        self.gm = f'Gross Margin'

        # To avoid double counting growth, for growth-adjusted multiples
        # we take the forward growth rate with the current revenue multiple
        rev_mult = 'LTM' if type == 'NTM' else '2020'
        self.growth_adj_mult = f'Growth adjusted EV / {rev_mult} Revenue'
        self.df[self.growth_adj_mult] = self.df[f'EV / {rev_mult} Revenue'] / self.df[self.rev_g]
        return self

    def get_y_metric_list(self):
        return [self.rev_mult, self.gp_mult, self.growth_adj_mult]

    def get_x_metric_list(self):
        return self.df.select_dtypes(['float', 'int']).columns

    def to_frame(self):
        return self.df

    @st.cache(suppress_st_warning=True,
              hash_funcs={RegressionInput: reg_input_hash, RegressionOutput: reg_output_hash})
    def _regression(self, reg_input):
        df = reg_input.df
        reg_x_vars = reg_input.x_vars
        reg_y = reg_input.y_var

        if not reg_x_vars:
            return None

        df_r = df[[reg_y] + reg_x_vars].dropna()

        # Run the regression
        X = df_r[reg_x_vars]
        X = sm.add_constant(X)

        model = sm.OLS(df_r[reg_y], X).fit()

        # Compute Variance Inflation Factors
        df_v = df_r[reg_x_vars]
        vif_data = None
        if len(df_v.columns) >= 2:
            # VIF dataframe
            vif_data = pd.DataFrame()

            vif_data["feature"] = df_v.columns
            # calculating VIF for each feature
            vif_data["VIF"] = [variance_inflation_factor(df_v.values, i)
                               for i in range(len(df_v.columns))]

        # pvalue dataframe
        df_pvalues = model.params.to_frame().reset_index().rename(columns={'index': 'vars', 0: 'Beta'})
        df_pvalues['p-value'] = model.pvalues.to_frame().reset_index().rename(columns={0: 'p-value'})['p-value']
        df_pvalues['Statistical Significance'] = 'Low'
        df_pvalues.loc[df_pvalues['p-value'] <= 0.05, 'Statistical Significance'] = 'High'
        df_pvalues = df_pvalues[df_pvalues['vars'] != 'const']

        return RegressionOutput(reg_input, df_r, model, df_pvalues, vif_data)

    def regression(self, reg_x_vars, reg_y_var):
        self.reg_input = RegressionInput(self.df, reg_x_vars, reg_y_var)
        self.reg_output = self._regression(self.reg_input)
        return self

    def print(self, show_detail=False):
        # Print regression equation
        st.latex(self.reg_output.eq_str)

        def highlight_significant_rows(val):
            color = 'green' if val['p-value'] <= 0.05 else 'red'
            return [f"color: {color}"] * len(val)

        st.subheader("Summary", anchor='summary')
        st.write(f"1. N = {len(self.reg_output.df_r)} companies")

        # Assess model fit
        if self.reg_output.model.rsquared * 100 > 30:
            st.write(f"2. Model fit is **good** R ^ 2 = {self.reg_output.model.rsquared * 100: .2f}%")
            if self.reg_output.model.f_pvalue < 0.05:
                st.write(f"3. Model is **statistically significant** (F-test = {self.reg_output.model.f_pvalue:.2f})")
            else:
                st.write(
                    f"3. The regression is **NOT statistically significant** (F-test = {self.reg_output.model.f_pvalue:.2f})")

        else:
            st.write(f"2. Model fit is **poor** (R ^ 2 = {self.reg_output.model.rsquared * 100: .2f}%)")

        # Check for Multicolinearity
        if (
                self.reg_output.vif_data is not None
                and len(self.reg_output.vif_data[self.reg_output.vif_data['VIF'] > 10]) > 0
        ):
            st.write("4. **Potential multicolinearity**")
        else:
            st.write("4. **NO multicolinearity**")

        # print p-values
        st.write('***')
        for _, row in self.reg_output.df_pvalues.iterrows():
            str = 'strong' if row['Statistical Significance'] == 'High' else 'weak'
            st.write(f"* There is a **{str} relationship** between *'{self.reg_input.y_var}'* and *'{row['vars']}'*")

        st.table(self.reg_output.df_pvalues.set_index('vars').style.apply(highlight_significant_rows, axis=1))

        if show_detail:
            # Show details
            st.subheader("Details:", anchor='details')
            # Plot residuals
            for k, f in self.reg_output.plot_figs.items():
                st.write(f"Plotting residuals for **{k}**")
                st.pyplot(f)
                # st.pyplot(f)
            st.markdown('***')
            st.write(self.reg_output.model.summary())
            st.markdown('***')
            if self.reg_output.vif_data is not None:
                st.write("Variance Inflation Factors")
                st.table(self.reg_output.vif_data.set_index('feature'))

        return self


def workbench(show_detail):
    fwd_time = st.sidebar.selectbox('Timeline', ('2021', 'NTM'))
    slice_by_growth = st.sidebar.radio("B2B SaaS Dataset", ['High growth only', 'All', 'All (excl. high growth)'])

    e = Experiment().set_fwd_timeline(fwd_time).filter(slice_by_growth)

    st.sidebar.write("**Regression:**")
    y_sel = st.sidebar.radio("Target metric", e.get_y_metric_list())
    st.sidebar.text("Select independent variable(s)")

    st.header("Regression")
    # Check if user selected revenue growth and/or gross margin
    reg_x_cols = [i for i in [e.rev_g, e.gm] if st.sidebar.checkbox(i, value={e.rev_g: True, e.gm: True}, key=i)]
    remaining_cols = list(set(e.get_x_metric_list()) - {e.rev_g, e.gm})
    reg_x_cols += st.sidebar.multiselect("Additional independent variables:", remaining_cols)
    e.regression(reg_x_vars=reg_x_cols, reg_y_var=y_sel).print(show_detail)


    ## Plots
    #st.header("Plots")
    #for _, x in zip(range(4), e.reg_input.x_vars):
    #    st.plotly_chart(get_scatter_fig(e.to_frame(), x=x, y=e.reg_input.y_var))
    #st.plotly_chart(get_scatter_fig(e.to_frame(), x=e.gm, y=e.reg_input.y_var))

    st.subheader("Dataset")
    st.beta_expander('Table Output') \
        .table(e.to_frame()[['Name'] + [y_sel] + reg_x_cols]
               .set_index('Name')
               .sort_values(y_sel, ascending=False))
    st.beta_expander('Full Raw Table Output').table(e.df_main)
    st.sidebar.info(f"""*{len(e.df)} companies selected*    
        *Prices as of {file_date}*""")

    return


def summary(e1, e2, e3, e4):
    st.header("High Growth B2B SaaS")
    st.markdown("""
    For high growth B2B SaaS, ***revenue growth*** (*not profitability*) ***drives valuation***
    * *Valuation multiples* are well explained by *revenue growth* 
        * Model fit is good (High R^2)
        * Revenue growth is a statistically significant factor (low p-value)
    * *Gross Margin* does not influence *valuation multiples*
        * Poor relationship between Revenue multiples and Gross margin (high p-value)
    """)
    with st.beta_expander("More info"):
        e1.print(True)

    st.markdown("""
        * Looking at Free Cash Flow % instead of Gross Margin yield similar results
            * Model fit is good (High R^2)
            * *Revenue growth* is a statistically significant factor (low p-value)
        * *FCF Margin* does not influence *valuation multiples*
            * Poor relationship between Revenue multiples and FCF margin (high p-value)
        """)
    with st.beta_expander("More info"):
        e2.print(True)

    st.markdown('***')
    st.header("B2B SaaS (excluding high growth)")
    st.markdown("""
        
        For the rest of B2B SaaS (i.e non high growth SaaS), the picture is less clear
        * *Revenue growth* by itself doesn't adequately explain *valuation multiples* 
            * Model fit is poor (low R^2)
        * But *Revenue growth* is still a statistically significant factor (low p-value)
        * *Gross Margin* does not influence *valuation multiples*
            * Poor relationship between Revenue multiples and Gross margin (high p-value)
        """)
    with st.beta_expander("More info"):
        e3.print(True)
    st.markdown("""
            * Looking at Free Cash Flow % instead of Gross Margin improves model fit
            * FCF Margin* has a **small positive effect** on *valuation multiples*
                * Low p-value but small Beta.
            * But overall *revenue growth* still has a much **larger effect** on valuation multiples than profitability
                * Low p-value and higher Beta relative to FCF %
            """)
    with st.beta_expander("More info"):
        e4.print(True)
    return


def main():
    #st.set_page_config(initial_sidebar_state="collapsed")
    sel = st.sidebar.radio("Menu", ['Summary', 'Workbench'])
    st.title('Impact of Growth and Margins on Valuation')
    show_detail = True
    # st.sidebar.checkbox('Show Details')

    # pre compute three experiments
    # Experiment 1
    e1 = Experiment() \
        .set_fwd_timeline('2021') \
        .filter('High growth only')
    e1.regression(reg_x_vars=[e1.rev_g, e1.gm], reg_y_var=e1.rev_mult)

    # Experiment 2
    e2 = Experiment() \
        .set_fwd_timeline('2021') \
        .filter('High growth only')
    e2.regression(reg_x_vars=[e2.rev_g, 'LTM FCF %'], reg_y_var=e2.rev_mult)

    # Experiment 3
    e3 = Experiment() \
        .set_fwd_timeline('2021') \
        .filter('All (excl. high growth)')
    e3.regression(reg_x_vars=[e3.rev_g, e3.gm], reg_y_var=e3.rev_mult)

    # Experiment 4
    e4 = Experiment() \
        .set_fwd_timeline('2021') \
        .filter('All (excl. high growth)')
    e4.regression(reg_x_vars=[e4.rev_g, 'LTM FCF %'], reg_y_var=e4.rev_mult)

    if sel == 'Workbench':
        return workbench(True)
    else:
        return summary(e1, e2, e3, e4)


if __name__ == "__main__":
    main()
