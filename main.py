import streamlit as st
import pandas as pd
import networkx as nx

st.title("지연 시뮬레이션 프로그램")

# ES(이른 시작), EF(이른 완료) 계산
def compute_es_ef(G):
    es = {}
    ef = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            es[node] = max(ef[p] for p in preds)
        else:
            es[node] = 0
        ef[node] = es[node] + G.nodes[node]['duration']
    return es, ef

# LS(늦은 시작), LF(늦은 완료) 계산
def compute_ls_lf(G, ef):
    lf = {}
    ls = {}
    for node in reversed(list(nx.topological_sort(G))):
        succs = list(G.successors(node))
        if succs:
            lf[node] = min(ls[s] for s in succs)
        else:
            lf[node] = ef[node]
        ls[node] = lf[node] - G.nodes[node]['duration']
    return ls, lf

# 여유시간(Float) 계산
def compute_float(es, ls):
    return {n: ls[n] - es[n] for n in es}

# 노드 레벨(단계) 계산
def assign_levels(G):
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if preds:
            levels[node] = max(levels[p] for p in preds) + 1
        else:
            levels[node] = 0
    return levels

# 네트워크(그래프) 생성
def build_network(df):
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

# CPM 계산 및 주경로(critical path) 추출
def compute_cpm(G):
    es, ef = compute_es_ef(G)
    ls, lf = compute_ls_lf(G, ef)
    float_ = compute_float(es, ls)
    def find_critical_paths(G, es, ef, ls, lf):
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
    critical_paths = find_critical_paths(G, es, ef, ls, lf)
    return es, ef, ls, lf, float_, critical_paths

# 네트워크 시각화
def visualize_network(G, es, ef, ls, lf, critical_paths, levels, title=None):
    import graphviz
    dot = graphviz.Digraph(format='png')
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
        if n in critical_path_nodes:
            table_bg = 'mistyrose'
            border_color = 'red'
        else:
            table_bg = 'white'
            border_color = 'black'
        label = f'''<
        <TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0" ALIGN="CENTER" BGCOLOR="{table_bg}">
          <TR>
            <TD FIXEDSIZE="TRUE" WIDTH="40" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE">{es[n]}</TD>
            <TD FIXEDSIZE="TRUE" WIDTH="40" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE">{ef[n]}</TD>
          </TR>
          <TR>
            <TD COLSPAN="2" FIXEDSIZE="TRUE" WIDTH="80" HEIGHT="24" ALIGN="CENTER" VALIGN="MIDDLE"><B>{G.nodes[n]['name']}({G.nodes[n]['duration']})</B></TD>
          </TR>
          <TR>
            <TD FIXEDSIZE="TRUE" WIDTH="40" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE">{ls[n]}</TD>
            <TD FIXEDSIZE="TRUE" WIDTH="40" HEIGHT="20" ALIGN="CENTER" VALIGN="MIDDLE">{lf[n]}</TD>
          </TR>
        </TABLE>
        >'''
        node_kwargs = {
            "label": label,
            "shape": "plaintext",
            "style": "filled",
            "fillcolor": "white",
            "color": border_color,
            "fontname": "Malgun Gothic"
        }
        dot.node(n, **node_kwargs)
    for l, nodes in level_nodes.items():
        if nodes:
            rank_line = '{rank=same; ' + ' '.join(nodes) + '}'
            dot.body.append(rank_line)
    if start_nodes:
        dot.body.append('{rank=source; ' + ' '.join(start_nodes) + '}')
    if end_nodes:
        dot.body.append('{rank=sink; ' + ' '.join(end_nodes) + '}')
    for u, v in G.edges:
        if (u, v) in critical_path_edges:
            dot.edge(u, v, color='red', penwidth='2.5')
        else:
            dot.edge(u, v, color='gray', penwidth='1.2')
    return dot

uploaded_file = st.file_uploader(
    "엑셀 공정표를 업로드하세요\n(필수 컬럼: ID, 공정명, 선행ID, 기간, 최소공사일, 단축 단가 (원/일), 연장 단가 (원/일))",
    type=["xlsx", "xls"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["ID"] = df["ID"].astype(str)
    df["선행ID"] = df["선행ID"].astype(str)
    df["단축 가능 일수"] = df["기간"] - df["최소공사일"]
    st.write("업로드된 공정표:")
    st.dataframe(df)

    G = build_network(df)
    es, ef, ls, lf, float_, critical_paths = compute_cpm(G)
    levels = assign_levels(G)
    dot = visualize_network(G, es, ef, ls, lf, critical_paths, levels)
    st.subheader("AON 네트워크")
    st.graphviz_chart(dot.source)

    st.subheader("지연 시뮬레이션")
    selected_task = st.selectbox("지연 공정을 선택하세요", df["ID"])
    delay_days = st.number_input("지연 일수 입력", min_value=1, max_value=30, value=3)
    delay_cost = G.nodes[selected_task]["delay_cost"] * delay_days

    if st.button("지연 반영하여 네트워크 재계산/시각화"):
        df_sim = df.copy()
        df_sim.loc[df_sim["ID"] == selected_task, "기간"] += delay_days
        df_sim["단축 가능 일수"] = df_sim["기간"] - df_sim["최소공사일"]
        G_sim = build_network(df_sim)
        es_sim, ef_sim, ls_sim, lf_sim, float_sim, critical_paths_sim = compute_cpm(G_sim)
        levels_sim = assign_levels(G_sim)
        dot_sim = visualize_network(G_sim, es_sim, ef_sim, ls_sim, lf_sim, critical_paths_sim, levels_sim)
        st.subheader("지연 반영된 AON 네트워크 (시뮬레이션 결과)")
        st.graphviz_chart(dot_sim.source)

        total_duration_before = max(ef.values())
        total_duration_after = max(ef_sim.values())
        delay_days_total = total_duration_after - total_duration_before

        st.write(f"전체 공사 기간: {total_duration_before}일 → {total_duration_after}일 (총 {delay_days_total}일 지연)")

        if delay_days_total > 0:
            # 모든 CP 경로별로, 지연된 공정 이후의 단축 가능 일수 계산
            reducible_days_list = []
            for path in critical_paths_sim:
                if selected_task in path:
                    idx = path.index(selected_task)
                    after_nodes = path[idx+1:]
                    reducible_days_list.append(sum([G_sim.nodes[n]['max_reduction'] for n in after_nodes]))
                else:
                    reducible_days_list.append(sum([G_sim.nodes[n]['max_reduction'] for n in path]))
            min_reducible_days = min(reducible_days_list) if reducible_days_list else 0
            X = min(delay_days_total, min_reducible_days)
            st.write(f"단축 가능 최대 일수: {X}일 (지연일수: {delay_days_total}일, 모든 CP별 단축 가능 일수: {reducible_days_list})")
        else:
            st.write("지연이 발생하지 않았으므로, 단축 시나리오가 필요하지 않습니다.")
