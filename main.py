import streamlit as st
import pandas as pd
import networkx as nx
import graphviz
import plotly.graph_objects as go
import io
import datetime

st.set_page_config(page_title="C-Fit", layout="wide")
st.title("공정  지연 시뮬레이션 및 대응 시스템")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stNumberInput, 
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.3rem;
    }
    section[data-testid="stSidebar"] .stSubheader {
        margin-bottom: 0.5rem;
        margin-top: 0.7rem;
    }
    </style>
""", unsafe_allow_html=True)

# 히스토리 초기화 및 표시
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'selected_history_idx' not in st.session_state:
    st.session_state['selected_history_idx'] = None
with st.sidebar:
    st.subheader("🕑 이전 분석 결과")
    history = st.session_state['history']
    if history:
        for idx, item in enumerate(history):
            label = f"{item['desc']}"
            if st.button(label, key=f"history_{idx}"):
                st.session_state['selected_history_idx'] = idx
        if st.button("히스토리 전체 삭제", key="clear_history"):
            st.session_state['history'] = []
            st.session_state['selected_history_idx'] = None
    else:
        st.caption("저장된 분석 결과가 없습니다.")
selected_idx = st.session_state.get('selected_history_idx')
history = st.session_state['history']
if selected_idx is not None and selected_idx < len(history):
    st.info(f"이전 분석 결과 (저장 시각: {history[selected_idx]['timestamp']})")
    st.write(history[selected_idx]['desc'])
    st.dataframe(history[selected_idx]['summary_df'], use_container_width=True)
    st.write("---")

# 사이드바 설정
with st.sidebar:
    st.subheader("💰 계약 정보")
    contract_amount = st.number_input("계약금 (원)", min_value=0, value=0, step=1000000, format="%d", help="프로젝트의 총 계약금액을 입력하세요")
    liquidated_damages_percent = st.number_input("지체상금 비율 (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%f", help="계약금 대비 지체상금 비율을 입력하세요")
    st.subheader("🎯 시뮬레이션 설정")
    default_delay_days = st.number_input("기본 지연일수", min_value=1, max_value=100, value=3, help="지연 공정 선택 시 기본으로 설정될 지연일수")
    max_delay_days = st.number_input("최대 지연일수", min_value=1, max_value=1000, value=100, help="지연 공정별 최대 입력 가능한 지연일수")
    st.subheader("ℹ️ 정보")
    st.info("""
    **C-Fit (Critical Path FITting System)**
    공정 지연 시뮬레이션 및 대응 시스템입니다.
    
    **사용법:**
    1. Excel 파일 업로드
    2. 지연 공정 선택
    3. 지연일수 입력
    4. 시뮬레이션 실행
    """)

def compute_cpm_times(G):
    """CPM 시간 계산 - 이른/늦은 시작/완료 시간, 여유시간"""
    # 이른 시작/완료 시간 계산
    es, ef = {}, {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        es[node] = max(ef[p] for p in preds) if preds else 0
        ef[node] = es[node] + G.nodes[node]['duration']
    
    # 늦은 시작/완료 시간 계산
    lf, ls = {}, {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        lf[node] = min(ls[s] for s in succs) if succs else ef[node]
        ls[node] = lf[node] - G.nodes[node]['duration']
    
    # 여유시간 계산
    float_ = {n: ls[n] - es[n] for n in es}
    
    return es, ef, ls, lf, float_

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
            delay_cost=int(row["지연 단가 (원/일)"]),
            max_reduction=int(row["단축 가능 일수"]))
        
        predecessor = row["선행ID"]
        if predecessor is not None and str(predecessor).strip() not in ('', '-'):
            for pre in str(predecessor).split(","):
                G.add_edge(str(pre.strip()), str(row["ID"]))
    return G

def compute_cpm(G):
    """CPM 계산 - 이른/늦은 시간, 여유시간, 주공정"""
    es, ef, ls, lf, float_ = compute_cpm_times(G)
    
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
    
    def get_network_nodes(G, levels):
        """네트워크 노드 정보 수집"""
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
        
        return start_nodes, end_nodes, level_nodes
    
    start_nodes, end_nodes, level_nodes = get_network_nodes(G, levels)
    
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

def analyze_network(data_df):
    """네트워크 분석"""
    G_network = build_network(data_df)
    es_network, ef_network, ls_network, lf_network, float_network, critical_paths_network = compute_cpm(G_network)
    levels_network = assign_levels(G_network)
    return G_network, es_network, ef_network, ls_network, lf_network, float_network, critical_paths_network, levels_network

def visualize_network_with_title(G, es, ef, ls, lf, critical_paths, levels, title):
    """네트워크 시각화 및 제목 표시"""
    dot_network = visualize_network(G, es, ef, ls, lf, critical_paths, levels)
    st.subheader(title)
    st.graphviz_chart(dot_network.source)

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
    def get_target_nodes():
        """단축 대상 노드들 결정"""
        if reduction_scope == "after_delay":
            return get_after_delay_nodes(critical_paths, selected_tasks)
        elif reduction_scope == "all_critical":
            return [get_all_critical_nodes(critical_paths)]
        else:
            return get_after_delay_nodes(critical_paths, selected_tasks)
    
    reducible_tasks = {}
    target_nodes_list = get_target_nodes()
    
    for target_nodes in target_nodes_list:
        for node in target_nodes:
            if G.nodes[node]['max_reduction'] > 0 and node not in reducible_tasks:
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
    
    def calculate_delay_costs(delay_info, G, delay_days_total, reduction_days, contract_amount, liquidated_damages_percent):
        """지연 관련 비용 계산"""
        delay_cost = sum(G.nodes[task_id]['delay_cost'] * delay_days 
                        for task_id, delay_days in delay_info.items() if delay_days > 0)
        total_delay_days = delay_days_total - reduction_days
        liquidated_damages = contract_amount * (liquidated_damages_percent / 100) * total_delay_days
        return delay_cost, total_delay_days, liquidated_damages
    
    delay_cost, total_delay_days, liquidated_damages = calculate_delay_costs(
        delay_info, G, delay_days_total, reduction_days, contract_amount, liquidated_damages_percent
    )
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
    st.write(f"**총 비용: {scenario['total_cost_with_delay']:,}원** (단축: {scenario['total_cost']:,}원, 지연: {scenario['delay_cost']:,}원, 지체상금: {scenario['liquidated_damages']:,}원)")
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
            '지연비용': f"{scenario['delay_cost']:,}원",
            '지체상금': f"{scenario['liquidated_damages']:,}원",
            '최종총비용': f"{scenario['total_cost_with_delay']:,}원",
            '총지연일수': f"{scenario['total_delay_days']}일",
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

def create_scenario_charts(scenarios, chart_height=400, show_grid=False):
    """시나리오 결과 차트 생성"""
    # 데이터 준비
    reduction_days = [s['reduction_days'] for s in scenarios]
    total_costs = [s['total_cost_with_delay'] for s in scenarios]
    reduction_costs = [s['total_cost'] for s in scenarios]
    delay_costs = [s['delay_cost'] for s in scenarios]
    liquidated_damages = [s['liquidated_damages'] for s in scenarios]
    
    # Y축 범위 계산 (데이터의 최소값부터 시작)
    min_total_cost = min(total_costs)
    max_total_cost = max(total_costs)
    y_range = max_total_cost - min_total_cost
    y_start = min_total_cost - y_range * 0.1  # 최소값보다 10% 아래부터 시작
    
    # 1. 총 비용 비교 차트
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=reduction_days,
        y=total_costs,
        name='총 비용',
        marker_color='lightcoral'
    ))
    fig1.update_layout(
        title='단축일수별 총 비용 비교',
        xaxis_title='단축일수 (일)',
        yaxis_title='비용 (원)',
        height=chart_height,
        yaxis=dict(
            range=[y_start, max_total_cost + y_range * 0.1],  # 데이터 범위에 맞게 조정
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    # 2. 비용 구성 차트
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=reduction_days,
        y=reduction_costs,
        name='단축 비용',
        marker_color='lightblue'
    ))
    fig2.add_trace(go.Bar(
        x=reduction_days,
        y=delay_costs,
        name='지연 비용',
        marker_color='lightgreen'
    ))
    fig2.add_trace(go.Bar(
        x=reduction_days,
        y=liquidated_damages,
        name='지체상금',
        marker_color='pink'
    ))
    fig2.update_layout(
        title='비용 구성 분석',
        xaxis_title='단축일수 (일)',
        yaxis_title='비용 (원)',
        barmode='stack',
        height=chart_height,
        yaxis=dict(
            rangemode='tozero',  # 0부터 시작 (스택 바 차트이므로)
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=show_grid,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    return fig1, fig2

def export_to_excel(scenarios, best_scenario, df, delay_info):
    """Excel 파일로 내보내기"""
    # 임시 파일을 사용하여 BytesIO 호환성 문제 해결
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
            # 시나리오 요약
            summary_data = []
            for scenario in scenarios:
                summary_data.append({
                    '단축일수': scenario['reduction_days'],
                    '단축비용': scenario['total_cost'],
                    '지연비용': scenario['delay_cost'],
                    '지체상금': scenario['liquidated_damages'],
                    '최종총비용': scenario['total_cost_with_delay'],
                    '총지연일수': scenario['total_delay_days']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='시나리오_요약', index=False)
            
            # 최적 시나리오 상세
            if best_scenario['tasks']:
                best_tasks_df = pd.DataFrame(best_scenario['tasks'])
                best_tasks_df.to_excel(writer, sheet_name='최적시나리오_상세', index=False)
            
            # 원본 데이터
            df.to_excel(writer, sheet_name='원본데이터', index=False)
            
            # 지연 정보
            delay_df = pd.DataFrame([
                {'공정ID': task_id, '지연일수': delay_days}
                for task_id, delay_days in delay_info.items()
            ])
            delay_df.to_excel(writer, sheet_name='지연정보', index=False)
    
    # 파일을 읽어서 BytesIO로 반환
    with open(tmp_file.name, 'rb') as f:
        output = io.BytesIO(f.read())
    
    # 임시 파일 삭제
    os.unlink(tmp_file.name)
    
    output.seek(0)
    return output





uploaded_file = st.file_uploader(
    "공정표 생성을 위한 데이터 파일을 업로드하세요.\n\n(필수 컬럼: ID, 공정명, 선행ID, 기간, 최소공사일, 단축 단가 (원/일), 지연 단가 (원/일))",
    type=["xlsx", "xls"]
)

def validate_data(df):
    required_columns = ["ID", "공정명", "선행ID", "기간", "최소공사일", "단축 단가 (원/일)", "지연 단가 (원/일)"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
    for col in ["기간", "최소공사일", "단축 단가 (원/일)", "지연 단가 (원/일)"]:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if (df[col] < 0).any():
            raise ValueError(f"{col}은 0 이상이어야 합니다.")
    if (df["최소공사일"] > df["기간"]).any():
        raise ValueError("최소공사일은 기간보다 클 수 없습니다.")
    return True

def check_network_validity(G):
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            raise ValueError(f"순환 참조가 발견되었습니다: {' → '.join([str(node) for node in cycles[0]])}")
    except nx.NetworkXNoCycle:
        pass
    if not nx.is_weakly_connected(G):
        st.warning("네트워크가 완전히 연결되지 않았습니다. 일부 공정이 고립되어 있을 수 있습니다.")
    return True

def process_uploaded_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        validate_data(df)
        df["ID"] = df["ID"].astype(str)
        df["선행ID"] = df["선행ID"].astype(str)
        df["단축 가능 일수"] = df["기간"] - df["최소공사일"]
        G = build_network(df)
        check_network_validity(G)
        return df, G
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        return None, None

def get_delay_info(selected_tasks, df):
    delay_info = {}
    if selected_tasks:
        st.markdown('<p style="font-size: 0.875rem; color: #6c757d; margin-bottom: 0.5rem;">각 공정별 지연일수를 입력하세요.</p>', unsafe_allow_html=True)
        for task_id in selected_tasks:
            task_name = df.loc[df["ID"] == task_id, "공정명"].iloc[0]
            delay_days = st.number_input(f"{task_id} - {task_name}", min_value=0, max_value=max_delay_days, value=default_delay_days, key=f"delay_{task_id}")
            delay_info[task_id] = delay_days
    return delay_info

if uploaded_file:
    df, G_initial = process_uploaded_data(uploaded_file)
    
    if df is not None and G_initial is not None:
        st.write("데이터:")
        st.dataframe(df)

        G, es, ef, ls, lf, float_, critical_paths, levels = analyze_network(df)
        visualize_network_with_title(G, es, ef, ls, lf, critical_paths, levels, "AON 네트워크 공정표")

        st.subheader("지연 시뮬레이션")
        selected_tasks = st.multiselect("지연 공정을 선택하세요. (여러 개 선택 가능)", df["ID"])

        delay_info = get_delay_info(selected_tasks, df)

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

        def create_simulation_data(df, delay_info):
            """시뮬레이션용 데이터 생성"""
            df_sim = df.copy()
            for task_id, delay_days in delay_info.items():
                if delay_days > 0:
                    df_sim.loc[df_sim["ID"] == task_id, "기간"] += delay_days
            df_sim["단축 가능 일수"] = df_sim["기간"] - df_sim["최소공사일"]
            return df_sim

        # 시뮬레이션 실행 조건: 버튼 클릭 + 지연 공정 선택 + delay_info 입력
        if run_simulation and selected_tasks and any(delay_info.values()):
            with st.spinner("시뮬레이션을 실행하고 있습니다..."):
                df_sim = create_simulation_data(df, delay_info)

                G_sim, es_sim, ef_sim, ls_sim, lf_sim, float_sim, critical_paths_sim, levels_sim = analyze_network(df_sim)
                visualize_network_with_title(G_sim, es_sim, ef_sim, ls_sim, lf_sim, critical_paths_sim, levels_sim, "반영된 AON 네트워크 공정표")

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

                    st.subheader("공정 단축 시나리오 분석")

                    reducible_tasks = find_reducible_tasks(G_sim, critical_paths_sim, selected_tasks, reduction_scope)

                    with st.spinner("시나리오를 계산하고 있습니다..."):
                        scenarios = []
                        for reduction_days in range(0, X + 1):
                            scenario = calculate_scenario_costs(
                                reduction_days, reducible_tasks, delay_info, G_sim, 
                                delay_days_total, contract_amount, liquidated_damages_percent
                            )
                            scenarios.append(scenario)

                        best_scenario = min(scenarios, key=lambda x: x['total_cost_with_delay'])
                        
                        # 차트 생성
                        fig1, fig2 = create_scenario_charts(scenarios)
                        
                        # 차트 표시
                        st.subheader("시나리오 분석 차트")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        st.subheader("시나리오 비교")

                        styled_df = create_summary_dataframe(scenarios, best_scenario)
                        st.dataframe(styled_df, use_container_width=True)

                        st.success(f"**추천: {best_scenario['reduction_days']}일 단축 시나리오**")
                        display_scenario_tasks(best_scenario, "공정별 단축 우선순위 및 비용 분석표")

                        # Excel 파일 미리 생성
                        try:
                            excel_output = export_to_excel(scenarios, best_scenario, df, delay_info)
                            excel_data = excel_output.getvalue()
                            excel_ready = True
                        except Exception as e:
                            st.error(f"Excel 파일 생성 중 오류가 발생했습니다: {str(e)}")
                            excel_ready = False

                        # 내보내기 기능
                        st.subheader("결과 내보내기")
                        
                        if excel_ready:
                            st.download_button(
                                label="Excel 파일 다운로드",
                                data=excel_data,
                                file_name="C-Fit_시나리오_분석.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        st.write("---")

                        for scenario in scenarios:
                            with st.expander(f"시나리오 {scenario['reduction_days']}일 단축 (총 지연일수: {scenario['total_delay_days']}일) - 총 비용: {scenario['total_cost_with_delay']:,}원"):
                                display_scenario_tasks(scenario, "공정별 단축 계획표")
                else:
                    st.write("지연이 발생하지 않았으므로, 단축 시나리오가 필요하지 않습니다.")
        elif run_simulation:
            st.info("지연 공정과 지연일수를 모두 입력해야 시뮬레이션이 실행됩니다.")

# 시뮬레이션 실행 후 결과를 히스토리에 저장
if (
    'run_simulation' in locals() and run_simulation and
    'selected_tasks' in locals() and selected_tasks and
    'delay_info' in locals() and any(delay_info.values()) and
    'scenarios' in locals()
):
    desc = f"지연공정: {', '.join(selected_tasks)} / 지연일수: {', '.join(str(delay_info[t]) for t in selected_tasks)}"
    summary_df = create_summary_dataframe(scenarios, best_scenario).data
    st.session_state['history'].append({
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'desc': desc,
        'summary_df': summary_df,
        'scenarios': scenarios,
        'best_scenario': best_scenario,
        'selected_tasks': selected_tasks.copy(),
        'delay_info': delay_info.copy(),
        'contract_amount': contract_amount,
        'liquidated_damages_percent': liquidated_damages_percent
    })
