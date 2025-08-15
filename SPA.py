import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_spectra_data(filepath):
    """通用光谱数据加载函数"""
    try:
        # 读取CSV文件，第一行作为列名
        df = pd.read_csv(filepath, encoding='utf-8', header=0)

        # 提取波长列名（从第三列开始）
        wavelength_columns = df.columns[2:]
        # 将波长字符串转换为数值（去掉" nm"单位）
        wavelengths = [float(col.replace(' nm', '')) for col in wavelength_columns]

        # 提取光谱数据
        spectra_data = df.iloc[:, 2:].values.astype(float)

        return wavelengths, spectra_data
    except Exception as e:
        print(f"加载文件 {os.path.basename(filepath)} 失败: {str(e)}")
        return None, None


# ================== 预处理函数 ==================
def savitzky_golay(data, window_size=15, poly_order=3):
    """Savitzky-Golay平滑滤波"""
    return savgol_filter(data,
                         window_length=window_size,
                         polyorder=poly_order,
                         axis=1,  # 对波段维度进行滤波
                         mode='nearest')


def snv_normalize(spectra):
    global_mean = np.mean(spectra)
    global_std = np.std(spectra)

    mean_row = np.mean(spectra, axis=1, keepdims=True)  # 每行（每个样本）的均值
    std_row = np.std(spectra, axis=1, keepdims=True)  # 每行（每个样本）的标准差
    std_row[std_row == 0] = 1e-8  # 防止除零错误（添加极小值保护）
    snv_data = (spectra - mean_row) / std_row

    normalized = snv_data * global_std + global_mean
    return normalized


def process_pipeline(data):
    """完整处理流程：SG平滑 → SNV标准化"""
    # 参数可调窗口大小
    window_size = 15  # 可根据光谱特征调整（需为奇数）
    smoothed = savitzky_golay(data, window_size=window_size)
    return snv_normalize(smoothed)


# ================== 可视化函数 ==================
def plot_spectra(wavelengths, processed_spectra, variety_labels, health_labels, varieties):
    """可视化所有光谱数据"""
    # 创建更大的图表
    plt.figure(figsize=(14, 8))

    # 使用viridis颜色映射
    n_varieties = len(varieties)
    colors = plt.cm.viridis(np.linspace(0, 1, n_varieties))

    # 绘制所有光谱
    for idx, spectrum in enumerate(processed_spectra):
        variety_idx = variety_labels[idx]
        health = health_labels[idx]

        # 设置颜色和线型
        color = colors[variety_idx]
        linestyle = '-' if health == 0 else '--'
        alpha = 0.8 if health == 0 else 0.6
        linewidth = 0.8 if health == 0 else 0.7

        # 绘制光谱
        plt.plot(wavelengths, spectrum,
                 color=color,
                 linestyle=linestyle,
                 linewidth=linewidth,
                 alpha=alpha)

    # 添加图例
    legend_handles = []
    for i, variety in enumerate(varieties):
        # 健康样本图例
        legend_handles.append(plt.Line2D([0], [0],
                                         color=colors[i],
                                         linestyle='-',
                                         linewidth=2,
                                         label=f'H: {variety}'))
        # 患病样本图例
        legend_handles.append(plt.Line2D([0], [0],
                                         color=colors[i],
                                         linestyle='--',
                                         linewidth=1.5,
                                         alpha=0.8,
                                         label=f'D: {variety}'))

    # 图表美化
    ax = plt.gca()

    # 保留上边框和右边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 设置边框线宽和颜色
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)  # 加粗边框线
        spine.set_color('gray')  # 设置边框颜色为灰色

    plt.xlabel('波长 (nm)', fontsize=30)
    plt.ylabel('反射率', fontsize=30)

    # 设置x轴范围为450-900nm
    plt.xticks(np.arange(450, 901, 50), fontsize=30)
    plt.xlim(450, 900)

    plt.yticks(fontsize=30)
    plt.grid(False)

    # 添加图例
    plt.legend(handles=legend_handles,
               loc='upper left',
               frameon=True,
               fontsize=16,
               bbox_to_anchor=(0.02, 0.95),
               borderaxespad=0.5,
               labelspacing=0.8,
               handlelength=2,
               ncol=2)  # 分两列显示

    plt.tight_layout()
    plt.show()



# ================== SPA特征选择函数 ==================
def spa_feature_selection(X, wavelengths, n_selected=7):
    """SPA特征选择算法 - 取消权重部分的标准版本"""
    n_samples, n_features = X.shape
    selected_bands = []
    remaining = list(range(n_features))

    # 选择第一个特征：列向量L2范数最大
    norms = np.linalg.norm(X, axis=0)
    first = np.argmax(norms)
    selected_bands.append(first)
    remaining.remove(first)

    for _ in range(1, n_selected):
        max_norm = -1
        best_feature = None
        S = X[:, selected_bands]

        # 处理奇异矩阵的情况，使用伪逆
        try:
            inv = np.linalg.inv(S.T @ S)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(S.T @ S)

        # 计算投影矩阵
        P = np.eye(n_samples) - S @ inv @ S.T

        # 遍历剩余特征找最大投影
        for j in remaining:
            x = X[:, j]
            v = P @ x
            current_norm = np.linalg.norm(v)  # 取消权重计算
            if current_norm > max_norm:
                max_norm = current_norm
                best_feature = j

        if best_feature is None:
            break
        selected_bands.append(best_feature)
        remaining.remove(best_feature)

    # 将波段索引转换为波长值
    selected_wavelengths = [wavelengths[i] for i in sorted(selected_bands)]

    print(f"SPA算法选择的波长: {selected_wavelengths}")

    return selected_wavelengths


def visualize_spa_features(wavelengths, processed_spectra, variety_labels, health_labels, varieties,
                           selected_wavelengths):
    """可视化SPA选择的特征波段 - 绘制所有400条光谱曲线"""
    # 创建更大的图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置颜色映射（与预处理可视化一致）
    n_varieties = len(varieties)
    colors = plt.cm.viridis(np.linspace(0, 1, n_varieties))

    # 绘制所有400条光谱曲线
    for idx, spectrum in enumerate(processed_spectra):
        variety_idx = variety_labels[idx]
        health = health_labels[idx]

        # 设置颜色和线型
        color = colors[variety_idx]
        linestyle = '-' if health == 0 else '--'  # 健康实线，患病虚线
        alpha = 0.4  # 适当降低透明度以便区分

        ax.plot(wavelengths, spectrum,
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=alpha)

    # 标记SPA选择的波段
    for wl in selected_wavelengths:
        # 找到最接近的波长索引
        idx = np.argmin(np.abs(np.array(wavelengths) - wl))

        # 绘制醒目的垂直线
        ax.axvline(wl, color='g', linestyle='--', linewidth=2, alpha=0.7)

        # 添加文本标注（放在顶部）
        ax.text(wl, ax.get_ylim()[1] * 0.5,
                f'{wl:.1f}nm',
                ha='center', va='bottom', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    # 创建图例
    legend_handles = []
    for i, variety in enumerate(varieties):
        # 健康样本图例
        legend_handles.append(plt.Line2D([0], [0],
                                         color=colors[i],
                                         linestyle='-',
                                         linewidth=2,
                                         label=f'H: {variety}'))
        # 患病样本图例
        legend_handles.append(plt.Line2D([0], [0],
                                         color=colors[i],
                                         linestyle='--',
                                         linewidth=1.5,
                                         label=f'D: {variety}'))

    # 添加图例
    plt.legend(handles=legend_handles,
               loc='upper left',
               frameon=True,
               fontsize=16,
               bbox_to_anchor=(0.02, 0.95),
               borderaxespad=0.5,
               labelspacing=0.8,
               handlelength=2,
               ncol=2)  # 分两列显示

    # 设置x轴范围为450-900nm
    plt.xticks(np.arange(450, 901, 50))
    plt.xlim(450, 900)
    ax.set_xlabel('wavelength (nm)', fontsize=30)
    ax.set_ylabel('reflectance', fontsize=30)
    ax.tick_params(labelsize=30)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 创建特征差异表格
    diff_data = []
    for wl in selected_wavelengths:
        idx = np.argmin(np.abs(np.array(wavelengths) - wl))
        h_values = processed_spectra[[i for i, h in enumerate(health_labels) if h == 0], idx]
        d_values = processed_spectra[[i for i, h in enumerate(health_labels) if h == 1], idx]
        h_mean = np.mean(h_values)
        d_mean = np.mean(d_values)
        diff = abs(h_mean - d_mean)
        diff_data.append([wl, h_mean, d_mean, diff])

    diff_df = pd.DataFrame(diff_data,
                           columns=['波长(nm)', '健康样本均值', '黄龙病样本均值', '差异'])

    # 打印表格
    print("\nSPA选择的特征波段差异分析:")
    print(diff_df.to_string(index=False))

    return fig, diff_df

# ================== 主程序 ==================
if __name__ == "__main__":
    # 数据集路径配置
    base_path = r'D:\CSV\对比组'

    # 定义柑橘品种列表
    varieties = ['wengzhoumigan', 'hongmeiren', 'penggan', 'tiancheng']
    n_varieties = len(varieties)

    # 存储所有平均光谱
    all_healthy_spectra = []
    all_disease_spectra = []
    wavelengths = None

    # 处理每个品种
    for i, variety in enumerate(varieties):
        variety_healthy_spectra = []  # 存储该品种的健康平均光谱
        variety_disease_spectra = []  # 存储该品种的患病平均光谱

        # 处理健康样本 (1-50)
        for j in range(1, 51):
            file_path = os.path.join(base_path, f"{variety}{j}.csv")
            wl, data = load_spectra_data(file_path)

            if data is not None and wl is not None:
                # 只保留450-900nm的波段
                mask = (np.array(wl) >= 450) & (np.array(wl) <= 900)
                wl = np.array(wl)[mask]
                data = data[:, mask]

                # 计算该文件的平均光谱
                file_mean = np.mean(data, axis=0)
                variety_healthy_spectra.append(file_mean)

                # 设置波长（只需要一次）
                if wavelengths is None:
                    wavelengths = wl

        # 处理患病样本 (1-50)
        for j in range(1, 51):
            file_path = os.path.join(base_path, f"{variety}-hlb{j}.csv")
            wl, data = load_spectra_data(file_path)

            if data is not None and wl is not None:
                # 只保留450-900nm的波段
                mask = (np.array(wl) >= 450) & (np.array(wl) <= 900)
                wl = np.array(wl)[mask]
                data = data[:, mask]

                # 计算该文件的平均光谱
                file_mean = np.mean(data, axis=0)
                variety_disease_spectra.append(file_mean)

        # 添加到总列表
        all_healthy_spectra.append(variety_healthy_spectra)
        all_disease_spectra.append(variety_disease_spectra)

    all_spectra = []
    health_labels = []  # 0=健康, 1=患病
    variety_labels = []  # 品种索引

    # 收集健康光谱
    for i in range(n_varieties):
        for spectrum in all_healthy_spectra[i]:
            all_spectra.append(spectrum)
            health_labels.append(0)
            variety_labels.append(i)

    # 收集患病光谱
    for i in range(n_varieties):
        for spectrum in all_disease_spectra[i]:
            all_spectra.append(spectrum)
            health_labels.append(1)
            variety_labels.append(i)

    all_spectra = np.array(all_spectra)
    processed_spectra = process_pipeline(all_spectra)

    # 调用可视化函数
    plot_spectra(wavelengths, processed_spectra, variety_labels, health_labels, varieties)

    # ================== SPA特征选择 ==================
    # 使用所有400条光谱数据进行SPA特征选择
    selected_wavelengths = spa_feature_selection(processed_spectra, wavelengths, n_selected=7)

    # 可视化SPA特征选择结果（传入所有必要参数）
    visualize_spa_features(wavelengths,
                           processed_spectra,
                           variety_labels,
                           health_labels,
                           varieties,
                           selected_wavelengths)