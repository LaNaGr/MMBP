import pandas as pd
import numpy as np
import itertools
import os


class Read_Recipes:
    """
    Read the recipes from p1X.txt and print them
    """
    def __init__(self, file="./Data_G/p11.txt"):
        """

        """
        chart1_data, chart2_data, chart3_data, num_machine_part1, num_stage = self.open_p_series(file)
        self.num_stage = num_stage
        self.chart1_df = self.chart1_to_df(chart1_data, num_machine_part1)
        self.chart2_df = self.chart2_to_df(chart2_data)
        self.chart3_df = self.chart3_to_df(chart3_data)

    @staticmethod
    def rebuild_from_to(df):
        """
        rebuild from_to column to refilled dataframe
        e.g. 1*50 -> 1, 2, 3, ..., 50
        """
        df['from'] = df['from_to'].apply(lambda x: int(x.split('*')[0]))
        df['to'] = df['from_to'].apply(lambda x: int(x.split('*')[1]) if '*' in x else np.nan)
        df = df.drop('from_to', axis=1)
        # initialize a new dataframe of final result
        max_recipe = int(max(df['from'].max(), df['to'].max()))
        df_final = pd.DataFrame(index=range(1, max_recipe+1))
        df_final['recipe']=df_final.index
        # fill the final dataframe
        df_part2 = df[df['to'].notnull()]
        df_final = pd.merge(df_final, df, how='left', left_on='recipe', right_on='from')
        for i in df_part2.index:
            row = df_part2.loc[i,:]
            update_pos_in_final = df_final[(df_final['recipe']<=row['to'])&(df_final['recipe']>row['from'])].index
            df_final.iloc[update_pos_in_final, 1::]=[row.tolist()]*len(update_pos_in_final)
        return df_final

    def chart1_to_df(self, chart1_data, num_machine_part1):
        """
        chart1_data: list, each element is a list of strings, need to change to int
        """
        # if chart contains many part, which is split by '+'
        if '+' in list(itertools.chain(*chart1_data)):  # if chart contains many part, which is split by '+'; flatten first
            # final result df would be in the list
            count_plus, new_df_content = 0, [[]]
            num = [num_machine_part1]
            head = [[i for i in range(1, num_machine_part1+1)]]
            for line in chart1_data:
                # split by '+'?
                if '+' in line:
                    count_plus += 1
                    new_df_content.append([])
                    head.append(list(map(int,line[1::])))   # convert str to int
                    num.append(len(line)-1)
                else:
                    new_df_content[count_plus].append(line)
            for i in range(count_plus+1):
                """DF1.drop(DF1.columns[0], axis=1, inplace=True) if DF1 else None"""
                DF1 = pd.DataFrame(new_df_content[i], columns=['from_to'] + head[i])
                # each DF1 has to be reindexed by from_to (which has mutiple same values in one line)
                DF2 = self.rebuild_from_to(DF1).drop(['from', 'to'], axis=1)
                if i == 0:
                    recipe_df = DF2
                else:
                    recipe_df = pd.merge(recipe_df, DF2, on='recipe')
            # result is recipe_df
        else:
            recipe_df = pd.DataFrame(chart1_data, columns=['recipe'] + [i for i in range(1, num_machine_part1 + 1)])
        # change to int: (now it is already done)
        print('#####\nchart1 dataframe\n', recipe_df)
        recipe_df.replace('-', '', inplace=True)
        return recipe_df

    def chart2_to_df(self, chart2_data):
        """
        chart2_data: list, each element is a list of strings, need to change to int
        """
        if len(chart2_data):
            for line in chart2_data:
                begin_point_of_the_stage, end_point_of_the_stage = line[0].split('*')
                chart2_data[chart2_data.index(line)] = [begin_point_of_the_stage, end_point_of_the_stage] + line[1:]
            df = pd.DataFrame(chart2_data, columns=['begin','end'] + [i for i in range(1,self.num_stage + 1)], index=None)
            df.insert(2,'stage',df.index + 1)
            df.drop(df.loc[:, 1::], axis=1, inplace=True)
            # change to int: (now it is already done)
            print('#####\nchart2 dataframe\n', df)
        else:
            df = pd.DataFrame()
        return df

    def chart3_to_df(self, chart3_data):
        """
        chart3_data: list, each element is a list of strings, need to change to int
        """
        if len(chart3_data):
            release, due = [], []
            release_date_0 = chart3_data[0][1]
            due_date_0 = chart3_data[1][1]
            release_1 = release_date_0.split(',')
            due_1 = due_date_0.split(',')
            for i in release_1:
                release.append(i.split())
            for j in due_1:
                due.append(j.split())
            df1 = pd.DataFrame(release, columns=['from_to', 'release'])
            df2 = pd.DataFrame(due,columns=['from_to', 'due'])

            # if there is *
            if '*' in list(itertools.chain(*release))[0]:
                df1 = self.rebuild_from_to(df1).drop(['from', 'to'], axis=1)
            if '*' in list(itertools.chain(*due))[0]:
                df2 = self.rebuild_from_to(df2).drop(['from', 'to'], axis=1)
            if 'from_to' in df1.columns:
                df1.rename(columns={'from_to':'recipe'}, inplace=True)
            if 'from_to' in df2.columns:
                df2.rename(columns={'from_to':'recipe'}, inplace=True)
            df1 = df1.astype({'recipe': 'str'})
            df2 = df2.astype({'recipe': 'str'})
            df = pd.merge(df1, df2, on='recipe')
            print('#####\nchart3 dataframe\n',df)
        else:
            df = pd.DataFrame()
        return df

    def open_p_series(self, file):
        """
        open p_series txt file and read 3 charts
        """
        with open(file, 'r') as f:
            flag = 0
            lines = f.readlines()
            ignore_name_row = True
            chart1, chart2, chart3 = False, False, False
            chart1_data, chart2_data, chart3_data = [], [], []
            num_stage = 'No stage information'
            for line in lines:
                # first line
                if ignore_name_row:
                    flag += 1
                    ignore_name_row = False
                    # decide which kind of information is going on
                    if flag<1000:
                        chart1 = True
                    elif 1000<=flag<2000:
                        chart2 = True
                        chart1 = False
                    else:
                        chart3 = True
                        chart1, chart2 = False, False
                elif line == '\n':
                    # start to change chart
                    flag = flag - flag % 1000   # to N*1000 for record chart position
                    flag += 1000
                    ignore_name_row = True

                # # # # # with former 2 part, we can find the sheet head

                elif flag == 1:
                    # how many machines / stages
                    chars = line.split('\t')
                    num_machine_part1 = len(chars)
                    # print('How many units?\n', chars,' -- ', len(chars), 'machines')
                    flag += 1
                elif flag == 1001:
                    chars = line.split('\t')
                    num_stage = len(chars) if chars is not [] else 'No stage information'
                    # print('How many stages?\n', chars, ' -- ', len(chars), 'stages')
                    flag += 1

                # # # # # other--record message
                else:
                    if chart1:
                        chart1_data.append(line.strip('\n').strip('').split('\t'))
                    elif chart2:
                        chart2_data.append(line.strip('\n').split('\t'))
                    elif chart3:
                        chart3_data.append(line.strip('\n').split('\t'))
                    flag += 1
                    # num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
                    """num_ope_biases.append(num_ope_bias)
                    # Detect information of this job and return the number of operations
                    num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
                    nums_ope.append(num_ope)
                    # nums_option = np.concatenate((nums_option, num_option))
                    opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))"""
                    flag += 1
            print('chart1_data:', chart1_data)
            print('chart2_data:', chart2_data)
            print('chart3_data:', chart3_data)
        return chart1_data, chart2_data, chart3_data, num_machine_part1, num_stage


# the next 2 function is for the latter class
def remove_all_item_equal_2_sth_from_list(list_x, x):
    """
    remove all the items equal to 2 from the list
    """
    # a = [1, 2, 3, 2, 4, 2, 5, 2]
    while x in list_x:
        list_x.remove(x)
    return list_x


def from_mmbp_to_fjsp(recipes):
    num_recipe = len(recipes.chart1_df.index)   # fjsp_instance[0][0]
    num_stage = recipes.num_stage
    recipe_time = recipes.chart1_df
    num_unit = len(recipe_time.columns) - 1     # fjsp_instance[0][1]
    recipe_stage = recipes.chart2_df
    basic_head = ['num_stage']

    """recipe_time.replace('', np.nan, inplace=True)
    recipe_time.astype(float, errors='ignore')
    value_matrix = recipe_time.iloc[:, 1::].copy()
    mean_time = np.mean(value_matrix.to_numpy()) # fjsp_instance[0][2]"""

    unit_stage_relation_list = []   # save the relationship between unit and stage
    stage_list = ['stage'+str(i) for i in range(1, num_stage+1)]  # deep copy the basic_head

    for i in range(1, num_stage+1):
        basic_head.append('stage'+str(i))
        begin = int(recipe_stage[recipe_stage['stage']==i].begin.values[0])
        end = int(recipe_stage[recipe_stage['stage']==i].end.values[0])
        mid_list = [recipe_num for recipe_num in range(begin, end+1)]
        basic_head = basic_head+ mid_list

        unit_stage_relation_list.append(mid_list)   # save the relationship between unit and stage

    print(basic_head)
    fjsp_df = pd.DataFrame(columns=basic_head, index=recipe_time.index)
    fjsp_df.update(recipe_time)
    fjsp_df.replace('', np.nan, inplace=True)
    fjsp_df.replace('-', np.nan, inplace=True)

    # iterate each recipe to make fjsp instance
    f_ins = []
    for i in fjsp_df.index:
        # print('recipe', i, end='\t')
        row = fjsp_df.loc[i,:].copy()
        step_count = 0  # when change the value, to fjsp instead of row
        for stage in range(num_stage):
            # not nan
            count_usable_unit = row[unit_stage_relation_list[stage]].count()
            row[stage_list[stage]] = count_usable_unit if count_usable_unit > 0 else np.nan
            # print('stage', stage+1, 'count_usable_unit', count_usable_unit, end='\t')
            step_count += 1 if count_usable_unit > 0 else 0
        row['num_stage'] = step_count
        fjsp_df.loc[i,:] = row

    # insert the No.Machine column and fill the value if it is not nan
    for col in fjsp_df.columns:
        if col in recipe_time.columns:
            new_col_name = str(col)+'_machine'
            fjsp_df.insert(fjsp_df.columns.get_loc(col), new_col_name, col)
            # delete value if it is nan in col
            fjsp_df[new_col_name][fjsp_df[col].isnull()] = np.nan

    # build fjsp instance
    fjsp_list = [[num_recipe, num_unit, 1]]    # 暂时没有算mean
    print('fjsp_instance:', fjsp_list[0])
    for i in fjsp_df.index:
        row = fjsp_df.loc[i,:].copy().astype(float, errors='ignore')
        row.dropna(inplace=True)
        # new_list = remove_all_item_equal_2_sth_from_list(row.tolist(), np.nan)
        fjsp_row = row.tolist()
        print(fjsp_row)
        fjsp_list.append(fjsp_row)

    return fjsp_list


# this is the final class
class record_fjsp_add:
    def __init__(self, x):
        """x is the name of primary file, e.g. x = 'p11' """
        self.name = x
        file = f"./MMBP/Data_G/{x}.txt"
        recipes = Read_Recipes(file)
        self.fjsp = from_mmbp_to_fjsp(recipes)
        self.recipe_date = recipes.chart3_df
        self.relation_unit_stage = recipes.chart2_df
        self.save_record()

    def save_record(self, source_fold='./MMBP'):
        """save in file"""
        # save information of fjsp version
        os.mkdir(source_fold+'/Data_G_FJSP_Version') if os.path.exists(source_fold+'/Data_G_FJSP_Version') is False else None
        with open(source_fold+f'/Data_G_FJSP_Version/{self.name}.fjs', 'w') as f:
            for line in self.fjsp:
                f.write(' '.join(map(str, line)) + '\n')
        # save release and due date
        os.mkdir(source_fold+'/Data_G_FJSP_Version_RnD') if os.path.exists(source_fold+'/Data_G_FJSP_Version_RnD') is False else None
        file_path = source_fold + f'/Data_G_FJSP_Version_RnD/{self.name}.csv'
        self.recipe_date.to_csv(file_path)
        # save relation - stage and unit
        os.mkdir(source_fold+'/Data_G_FJSP_Version_StageUnit') if os.path.exists(source_fold+'/Data_G_FJSP_Version_StageUnit') is False else None
        self.relation_unit_stage.to_csv(source_fold+'/Data_G_FJSP_Version_StageUnit/{self.name}.csv')



if __name__ == '__main__':
    """x = 'p15'
    record_fjsp_add(x)"""
    x = 'i07'
    record_fjsp_add(x)



