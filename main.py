import streamlit as st
import pandas as pd
import networkx as nx
import graphviz

st.title("지연 시뮬레이션 프로그램")

def compute_es_ef(G):
    """이른 시작/완료 시간 계산"""
    es, ef = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        es[node] = max(ef[p] for p in preds) if preds else 0
        ef[node] = es[node] + G.nodes[node]['duration']
    return es, ef

def compute_ls_lf(G, ef):
    """늦은 시작/완료 시간 계산"""
    lf, ls = {}, {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        lf[node] = min(ls[s] for s in succs) if succs else ef[node]
        ls[node] = lf[node] - G.nodes[node]['duration']
    return ls, lf

def compute_float(es, ls):
    """여유시간 계산"""
    return {n: ls[n] - es[n] for n in es}

def assign_levels(G):
    """공정 레벨 할당"""
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        levels[node] = max(levels[p] for p in preds) + 1 if preds else 0
    return levels

def build_network(df):
    """엑셀 데이터를 네트워크로 변환"""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(str(row["ID"]),
            name=row["공정명"],
            duration=int(row["기간"]),
            min_duration=int(row["최소공사일"]),
            reduction_cost=int(row["단축 단가 (원/일)"]),
            delay_cost=int(row["연장 단가 (원/일)"]),
            max_reduction=int(row["단축 가능 일수"]))
        
        predecessor = row["선행ID"]
        if predecessor is not None and str(predecessor).strip() not in ('', '-'):
            for pre in str(predecessor).split(","):
                G.add_edge(str(pre.strip()), str(row["ID"]))
    return G

def compute_cpm(G):
    """CPM 계산 - 이른/늦은 시간, 여유시간, 주공정"""
    es, ef = compute_es_ef(G)
    ls, lf = compute_ls_lf(G, ef)
    float_ = compute_float(es, ls)
    
    def find_critical_paths(G, es, ls):
        """주공정 찾기"""
        start_nodes = [n for n in G.nodes if G.in_degree[n] == 0]
        end_nodes = [n for n in G.nodes if G.out_degree[n] == 0]
        critical_paths = []
        
        def dfs(path):
            current = path[-1]
            if current in end_nodes:
                critical_paths.append(list(path))
                return
            for succ in G.successors(current):
                if (ls[succ] - es[succ]) == 0:
                    dfs(path + [succ])
        
        for start in start_nodes:
            if (ls[start] - es[start]) == 0:
                dfs([start])
        return critical_paths
    
    critical_paths = find_critical_paths(G, es, ls)
    return es, ef, ls, lf, float_, critical_paths

def visualize_network(G, es, ef, ls, lf, critical_paths, levels):
    """네트워크 시각화"""
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    
    critical_path_nodes = set()
    for path in critical_paths:
        critical_path_nodes.update(path)
    
    critical_path_edges = set()
    for path in critical_paths:
        for i in range(len(path) - 1):
            critical_path_edges.add((path[i], path[i+1]))
    
    start_nodes = [n for n in G.nodes if G.in_degree[n] == 0]
    end_nodes = [n for n in G.nodes if G.out_degree[n] == 0]
    max_level = max(levels.values()) if levels else 0
    
    level_nodes = {}
    for l in range(max_level + 1):
        nodes = [n for n in G.nodes if levels[n] == l]
        if nodes:
            max_out_deg = max([G.out_degree[n] for n in nodes])
            center_nodes = [n for n in nodes if G.out_degree[n] == max_out_deg]
            center_node = center_nodes[0]
            rest = [n for n in nodes if n != center_node]
            half = len(rest) // 2
            nodes_sorted = rest[:half] + [center_node] + rest[half:]
        else:
            nodes_sorted = []
        level_nodes[l] = nodes_sorted
    
    for n in G.nodes:
        is_critical = n in critical_path_nodes
        table_bg = 'mistyrose' if is_critical else 'white'
        border_color = 'red' if is_critical else 'black'
        
        label = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0" ALIGN="CENTER" BGCOLOR="{table_bg}" STYLE="font-size:10px;">
          <TR>
            <TD FIXEDSIZE="TRUE" WIDTH="60" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE" STYLE="font-size:10px;">{es[n]}</TD>
            <TD FIXEDSIZE="TRUE" WIDTH="60" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE" STYLE="font-size:10px;">{ef[n]}</TD>
          </TR>
          <TR>
            <TD COLSPAN="2" FIXEDSIZE="TRUE" WIDTH="120" HEIGHT="25" ALIGN="CENTER" VALIGN="MIDDLE" STYLE="font-size:12px; text-align:left;"><B>{G.nodes[n]['name']}({G.nodes[n]['duration']})</B></TD>
          </TR>
          <TR>
            <TD FIXEDSIZE="TRUE" WIDTH="60" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE" STYLE="font-size:10px;">{ls[n]}</TD>
            <TD FIXEDSIZE="TRUE" WIDTH="60" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE" STYLE="font-size:10px;">{lf[n]}</TD>
          </TR>
        </TABLE>
        >'''
        
        dot.node(n, label=label, shape="plaintext", style="filled", fillcolor="white", color=border_color, fontname="Malgun Gothic")
    
    for l, nodes in level_nodes.items():
        if nodes:
            rank_line = '{rank=same; ' + ' '.join(nodes) + '}'
            dot.body.append(rank_line)
    
    if start_nodes:
        dot.body.append('{rank=source; ' + ' '.join(start_nodes) + '}')
    if end_nodes:
        dot.body.append('{rank=sink; ' + ' '.join(end_nodes) + '}')
    
    for u, v in G.edges:
        is_critical_edge = (u, v) in critical_path_edges
        dot.edge(u, v, color='red' if is_critical_edge else 'gray', penwidth='2.5' if is_critical_edge else '1.2')
    
    return dot

def analyze_and_visualize_network(data_df, title, is_simulation=False):
    """네트워크 분석 및 시각화"""
    G_network = build_network(data_df)
    es_network, ef_network, ls_network, lf_network, float_network, critical_paths_network = compute_cpm(G_network)
    levels_network = assign_levels(G_network)
    dot_network = visualize_network(G_network, es_network, ef_network, ls_network, lf_network, critical_paths_network, levels_network)
    
    st.subheader(title)
    st.graphviz_chart(dot_network.source)
    
    return G_network, es_network, ef_network, ls_network, lf_network, float_network, critical_paths_network, levels_network

def get_after_delay_nodes(critical_paths, selected_tasks):
    """지연된 공정 이후의 공정들 찾기"""
    after_nodes_list = []
    for path in critical_paths:
        max_delay_idx = -1
        for task_id in selected_tasks:
            if task_id in path:
                idx = path.index(task_id)
                max_delay_idx = max(max_delay_idx, idx)
        
        if max_delay_idx >= 0:
            after_nodes = path[max_delay_idx+1:]
        else:
            after_nodes = path
        after_nodes_list.append(after_nodes)
    return after_nodes_list

def get_all_critical_nodes(critical_paths):
    """주공정의 모든 공정들 찾기"""
    all_nodes = set()
    for path in critical_paths:
        all_nodes.update(path)
    return list(all_nodes)

def find_reducible_tasks(G, critical_paths, selected_tasks, reduction_scope="after_delay"):
    """단축 가능한 공정들 찾기"""
    reducible_tasks = {}
    
    if reduction_scope == "after_delay":
        # 지연된 공정 이후의 공정들만 대상
        target_nodes_list = get_after_delay_nodes(critical_paths, selected_tasks)
    elif reduction_scope == "all_critical":
        # 주공정의 모든 공정 대상
        target_nodes_list = [get_all_critical_nodes(critical_paths)]
    else:
        # 기본값: 지연된 공정 이후의 공정들만
        target_nodes_list = get_after_delay_nodes(critical_paths, selected_tasks)
    
    for target_nodes in target_nodes_list:
        for node in target_nodes:
            if G.nodes[node]['max_reduction'] > 0:
                if node not in reducible_tasks:
                    reducible_tasks[node] = {
                        'name': G.nodes[node]['name'],
                        'max_reduction': G.nodes[node]['max_reduction'],
                        'reduction_cost': G.nodes[node]['reduction_cost'],
                        'current_duration': G.nodes[node]['duration']
                    }
    return reducible_tasks

def calculate_scenario_costs(reduction_days, reducible_tasks, delay_info, G, delay_days_total, contract_amount, liquidated_damages_percent):
    """시나리오 비용 계산"""
    remaining_days = reduction_days
    scenario_tasks = []
    total_cost = 0
    
    sorted_tasks = sorted(reducible_tasks.items(), key=lambda x: x[1]['reduction_cost'])
    
    for task_id, task_info in sorted_tasks:
        if remaining_days <= 0:
            break
        
        task_reduction = min(remaining_days, task_info['max_reduction'])
        
        if task_reduction > 0:
            task_cost = task_reduction * task_info['reduction_cost']
            scenario_tasks.append({
                'task_id': task_id,
                'task_name': task_info['name'],
                'reduction_days': task_reduction,
                'reduction_cost_per_day': task_info['reduction_cost'],
                'total_cost': task_cost
            })
            total_cost += task_cost
            remaining_days -= task_reduction
    
    delay_cost = 0
    for task_id, delay_days in delay_info.items():
        if delay_days > 0:
            delay_cost += G.nodes[task_id]['delay_cost'] * delay_days
    total_delay_days = delay_days_total - reduction_days
    liquidated_damages = contract_amount * (liquidated_damages_percent / 100) * total_delay_days
    total_cost_with_delay = total_cost + delay_cost + liquidated_damages
    
    return {
        'reduction_days': reduction_days,
        'tasks': scenario_tasks,
        'total_cost': total_cost,
        'delay_cost': delay_cost,
        'total_delay_days': total_delay_days,
        'liquidated_damages': liquidated_damages,
        'total_cost_with_delay': total_cost_with_delay
    }

def display_scenario_tasks(scenario, title):
    """시나리오 상세 정보 표시"""
    st.write(f"**총 비용: {scenario['total_cost_with_delay']:,}원** (단축: {scenario['total_cost']:,}원, 지연연장: {scenario['delay_cost']:,}원, 지체상금: {scenario['liquidated_damages']:,}원)")
    if scenario['tasks']:
        st.subheader(title)
        task_data = []
        for task in scenario['tasks']:
            task_data.append({
                '공정ID': task['task_id'],
                '공정명': task['task_name'],
                '단축일수': task['reduction_days'],
                '단축단가(원/일)': task['reduction_cost_per_day'],
                '총비용(원)': task['total_cost']
            })
        task_df = pd.DataFrame(task_data)
        st.dataframe(task_df, use_container_width=True)
    else:
        st.write("단축할 공정이 없습니다.")

def create_summary_dataframe(scenarios, best_scenario):
    """시나리오 비교표 생성"""
    summary_data = []
    for scenario in scenarios:
        summary_data.append({
            '단축일수': f"{scenario['reduction_days']}일",
            '단축비용': f"{scenario['total_cost']:,}원",
            '연장비용': f"{scenario['delay_cost']:,}원",
            '총지연일수': f"{scenario['total_delay_days']}일",
            '지체상금': f"{scenario['liquidated_damages']:,}원",
            '최종총비용': f"{scenario['total_cost_with_delay']:,}원",
            '정렬용비용': scenario['total_cost_with_delay']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('정렬용비용')
    summary_df = summary_df.drop('정렬용비용', axis=1)
    
    def highlight_recommended(row):
        try:
            current_cost = float(row['최종총비용'].replace('원', '').replace(',', ''))
            return ['background-color: lightgreen; color: black'] * len(row) if current_cost == best_scenario['total_cost_with_delay'] else [''] * len(row)
        except Exception:
            return [''] * len(row)
    
    styled_df = summary_df.style.apply(highlight_recommended, axis=1)
    return styled_df

uploaded_file = st.file_uploader(
    "엑셀을 업로드하세요.\n(필수 컬럼: ID, 공정명, 선행ID, 기간, 최소공사일, 단축 단가 (원/일), 연장 단가 (원/일))",
    type=["xlsx", "xls"]
)

# 계약금과 지체상금 입력 필드를 2열로 배치
col1, col2 = st.columns(2)
with col1:
    contract_amount = st.number_input("계약금을 입력하세요. (원)", min_value=0, value=0, step=1000000, format="%d")
with col2:
    liquidated_damages_percent = st.number_input("지체상금에 대한 비율을을 입력하세요. (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%f")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["ID"] = df["ID"].astype(str)
    df["선행ID"] = df["선행ID"].astype(str)
    df["단축 가능 일수"] = df["기간"] - df["최소공사일"]
    st.write("업로드된 데이터:")
    st.dataframe(df)

    G, es, ef, ls, lf, float_, critical_paths, levels = analyze_and_visualize_network(df, "AON 네트워크 공정표")

    st.subheader("지연 시뮬레이션")
    selected_tasks = st.multiselect("지연 공정을 선택하세요. (여러 개 선택 가능)", df["ID"])

    delay_info = {}
    if selected_tasks:
        st.write("각 공정별 지연일수를 입력하세요.:")
        for task_id in selected_tasks:
            task_name = df.loc[df["ID"] == task_id, "공정명"].iloc[0]
            delay_days = st.number_input(
                f"{task_id} - {task_name}", 
                min_value=0, 
                max_value=1000, 
                value=3,
                key=f"delay_{task_id}"
            )
            delay_info[task_id] = delay_days

    # 단축 대상 범위 선택 (항상 노출)
    reduction_options = [
        ("after_delay", "지연된 공정 이후의 공정들만"),
        ("all_critical", "주공정의 모든 공정")
    ]
    reduction_scope = st.selectbox(
        "단축 대상 범위 선택",
        options=reduction_options,
        format_func=lambda x: x[1],
        index=0,
        key="reduction_scope"
    )[0]

    # 시뮬레이션 실행 버튼 추가
    run_simulation = st.button("시뮬레이션 실행")

    # 시뮬레이션 실행 조건: 버튼 클릭 + 지연 공정 선택 + delay_info 입력
    if run_simulation and selected_tasks and any(delay_info.values()):
        df_sim = df.copy()
        for task_id, delay_days in delay_info.items():
            if delay_days > 0:
                df_sim.loc[df_sim["ID"] == task_id, "기간"] += delay_days
        df_sim["단축 가능 일수"] = df_sim["기간"] - df_sim["최소공사일"]

        G_sim, es_sim, ef_sim, ls_sim, lf_sim, float_sim, critical_paths_sim, levels_sim = analyze_and_visualize_network(
            df_sim, "반영된 AON 네트워크 공정표", is_simulation=True
        )

        total_duration_before = max(ef.values())
        total_duration_after = max(ef_sim.values())
        delay_days_total = total_duration_after - total_duration_before

        st.write(f"전체 공사 기간: {total_duration_before}일 → {total_duration_after}일 (총 {delay_days_total}일 지연)")

        if delay_days_total > 0:
            if reduction_scope == "after_delay":
                after_nodes_list = get_after_delay_nodes(critical_paths_sim, selected_tasks)
                reducible_days_list = [sum([G_sim.nodes[n]['max_reduction'] for n in after_nodes]) for after_nodes in after_nodes_list]
                scope_description = "지연된 공정 이후의 공정들"
            else:  # all_critical
                all_critical_nodes = get_all_critical_nodes(critical_paths_sim)
                reducible_days_list = [sum([G_sim.nodes[n]['max_reduction'] for n in all_critical_nodes])]
                scope_description = "주공정의 모든 공정"

            min_reducible_days = min(reducible_days_list) if reducible_days_list else 0
            X = min(delay_days_total, min_reducible_days)
            st.write(f"단축 대상: {scope_description}")
            st.write(f"단축 가능 최대 일수: {X}일 (지연일수: {delay_days_total}일, 모든 주공정별 단축 가능 일수: {', '.join(str(x) for x in reducible_days_list)})")

            st.subheader("단축 시나리오 분석")

            reducible_tasks = find_reducible_tasks(G_sim, critical_paths_sim, selected_tasks, reduction_scope)

            scenarios = []
            for reduction_days in range(0, X + 1):
                scenario = calculate_scenario_costs(
                    reduction_days, reducible_tasks, delay_info, G_sim, 
                    delay_days_total, contract_amount, liquidated_damages_percent
                )
                scenarios.append(scenario)

            best_scenario = min(scenarios, key=lambda x: x['total_cost_with_delay'])
            st.subheader("시나리오 비교표")

            styled_df = create_summary_dataframe(scenarios, best_scenario)
            st.dataframe(styled_df, use_container_width=True)

            st.success(f"**추천: {best_scenario['reduction_days']}일 단축 시나리오**")
            display_scenario_tasks(best_scenario, "최적 시나리오 단축 공정표")

            st.write("---")

            for scenario in scenarios:
                with st.expander(f"시나리오 {scenario['reduction_days']}일 단축 (총 지연일수: {scenario['total_delay_days']}일) - 총 비용: {scenario['total_cost_with_delay']:,}원"):
                    display_scenario_tasks(scenario, "단축 공정표")
        else:
            st.write("지연이 발생하지 않았으므로, 단축 시나리오가 필요하지 않습니다.")
    elif run_simulation:
        st.info("지연 공정과 지연일수를 모두 입력해야 시뮬레이션이 실행됩니다.")
