from heapq import merge
from logging import exception
from pprint import pprint
import gradio as gr
import gc
import re
import safetensors.torch
import os
import shutil
import os.path
import argparse
import modules.ui
import modules.ui_components as uc
import scripts.styles as sc
import csv
from modules import sd_models,script_callbacks,scripts, shared,sd_hijack,devices,sd_vae
from modules.ui import create_output_panel
from modules.shared import opts
from modules.sd_models import checkpoints_loaded
import scripts.merger.lora_merger as pluslora

from scripts.merger.bw_merger import TYPESEG,rwmergelog,freezemtime, umergegen
from scripts.merger.each_merger import TYPESEG, eachmerge,rwmergelog,freezemtime, eachmergegen
from scripts.merge_utils.xyplot import freezetime,numaker,numanager,nulister
from scripts.merge_utils.utils import savemodel
from scripts.merge_utils.imagen import uimggen

gensets=argparse.Namespace()
system_paste_fields = []
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
next_id_symbol = ">"
prev_id_symbol = "<"
last_id_symbol = "<<"
first_id_symbol = ">>"


stylepath = "./umerge_style/style.csv"
path_root = scripts.basedir()

def on_ui_train_tabs(params):
    txt2img_preview_params=params.txt2img_preview_params
    gensets.txt2img_preview_params=txt2img_preview_params
    return None
def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = uc.ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button

def load_historyf(id):
    filepath = os.path.join(path_root, "mergehistory.csv")

    id = int(id)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ii = row.get("ID", "")
            if int(ii) == id:
                row = row
        id = row.get("ID", "")
        Time = row.get("time", "")
        Name = row.get("name", "")
        ModelA = row.get("model A", "")
        ModelB = row.get("model B", "")
        ModelC = row.get("model C", "")
        WeightsMBW = row.get("WeightsMBW", "")
        BaseMBW = row.get("BaseMBW", "")
        WeightsA = row.get("WeightsA", "")
        WeightsB = row.get("WeightsB", "")
        BaseEach = row.get("BaseEach", "")
        Mode = row.get("mode", "")
        UseMBW = row.get("Use MBW", "")
        UseEach = row.get("Use Each", "")
        PlusLora = row.get("plus lora", "")
        CustomName = row.get("custum name", "")
        SaveSetting = row.get("save setting", "")
        UseID = row.get("UseID", "")
        SaveSetting = row.get("SaveSetting", "")
        cv = [id , Time , Name]
        mv = [ModelA , ModelB , ModelC]
        mbv = [WeightsMBW , BaseMBW]
        ev = [WeightsA , WeightsB , BaseEach]
        pr = [Mode , UseMBW , UseEach , PlusLora , CustomName , SaveSetting , UseID]
        core = gr.DataFrame.update(value= [cv])
        model = gr.DataFrame.update(value= [mv])
        mbw = gr.DataFrame.update(value= [mbv])
        each = gr.DataFrame.update(value= [ev])
        props = gr.DataFrame.update(value= [pr])
        return core, model, mbw, each, props
def onIDchange(mode:str, id):
    mlist = []
    filepath = os.path.join(path_root, "mergehistory.csv")
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        mlist = list(reader)
    id = int(id)
    if mode == "next":
        if id == len(mlist)-1:
            id = 0
        else:
            id += 1
    if mode == "prev":
        if id == 0:
            id = len(mlist)-1
        else:
            id -= 1
    if mode == "set":
        id = id
    if mode == "first":
        id = 1
    if mode == "last":
        id = len(mlist)-1
    return id



def on_ui_tabs():
    ps = sc.StyleDatabase(path = stylepath)
    weights_presets=""
    userfilepath = os.path.join(path_root, "scripts","mbwpresets.txt")
    if os.path.isfile(userfilepath):
        try:
            with open(userfilepath) as f:
                weights_presets = f.read()
                filepath = userfilepath
        except OSError as e:
                pass
    else:
        filepath = os.path.join(path_root, "scripts","mbwpresets_master.txt")
        try:
            with open(filepath) as f:
                weights_presets = f.read()
                shutil.copyfile(filepath, userfilepath)
        except OSError as e:
                pass
    # TODO: UIの改善とCSSの作成
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Merge", elem_id="tab_merge"):
            with gr.Row().style(equal_height=False):
                with gr.Column(scale=4):
                    #一応実装したけど、使うかどうかは知らん
                    gr.HTML(value="<p>Merge models and load it for generation</p>")
                    txt_multi_process_cmd = gr.TextArea(label="Multi Proc Cmd", visible=False, placeholder="Keep empty if dont use.")
                    radio_position_ids = gr.Radio(label="Skip/Reset CLIP position_ids", choices=["None", "Skip", "Force Reset"], value="None", visible=False, type="index")
                    #TODO モデルの大量マージとか気にならない？ あと、マージボードほしい。
                    with gr.Row():
                        #TODO 花札とモデルの検索機能。モジュールから拝借する。
                        model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model A",interactive=True)
                        create_refresh_button(model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                        model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model B",interactive=True)
                        create_refresh_button(model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                        model_c = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model C",interactive=True)
                        create_refresh_button(model_c, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                    mode = gr.Radio(label = "Merge Mode",choices = ["Weight sum:A*(1-alpha)+B*alpha", "Add difference:A+(B-C)*alpha",
                                                        "Triple sum:A*(1-alpha-beta)+B*alpha+C*beta",
                                                        "sum Twice:(A*(1-alpha)+B*alpha)*(1-beta)+C*beta",
                                                        ], value = "Weight sum:A*(1-alpha)+B*alpha")
                    # TODO: 絶対に計算方法まだあるでしょ。SINとかTANとか線形とか。なので探索する。
                    calcmode = gr.Radio(label = "Calcutation Mode",choices = ["normal", "cosineA", "cosineB", "smoothAdd","tensor"], value = "normal")
                    with gr.Row():
                        ub = gr.Radio(label="marge mode", interactive = True, choices=["None", "MBW", "Each"], value="None", type="index")
                        useeach = gr.Checkbox(label="use each", value=False, visible=False)
                        useblocks = gr.Checkbox(label="use blocks", value=False, visible=False)
                        base_alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                        base_beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=0.25)
                        #weights = gr.Textbox(label="weights,base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",lines=2,value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
                    with gr.Row():
                        merge = gr.Button(elem_id="model_merger_merge", value="Merge!",variant='primary')
                        mergeandgen = gr.Button(elem_id="model_merger_merge", value="Merge&Gen",variant='primary')
                        gen = gr.Button(elem_id="model_merger_merge", value="Gen",variant='primary')
                        stopmerge = gr.Button(elem_id="stopmerge", value="Stop",variant='primary')
                    #TODO: キャッシュからの保存
                    #TODO: 生成停止、スキップの追加
                    with gr.Row():
                        with gr.Column(scale = 4):
                            save_sets = gr.CheckboxGroup(["save model", "overwrite","safetensors","fp16","save metadata"], value=["safetensors"], label="save settings")
                        with gr.Column(scale = 2):
                            id_sets = gr.CheckboxGroup(["image", "PNG info"], label="write merged model ID to")
                    with gr.Row():
                        with gr.Column(min_width = 50, scale=2):
                            with gr.Row():
                                custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="model_converter_custom_name")
                                mergeid = gr.Textbox(label="merge from ID", elem_id="model_converter_custom_name",value = "-1")
                        with gr.Column(min_width = 50, scale=1):
                            with gr.Row():s_reverse= gr.Button(value="Set from ID(-1 for last)",variant='primary')
                    #TODO: i2i, GAN, script, Control net, inpaintへの対応
                    with gr.Accordion("Hires Fix , Batch size",open = False):
                        batch_size = denois_str = gr.Slider(minimum=0, maximum=8, step=1, label='Batch size', value=1, elem_id="sm_txt2img_batch_size")
                        hireson = gr.Checkbox(label = "hiresfix",value = False, visible = True,interactive=True)
                        with gr.Row(elem_id="txt2img_hires_fix_row1", variant="compact"):
                            hrupscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                            hr2ndsteps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                            denois_str = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")
                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                    hiresfix = [hireson,hrupscaler,hr2ndsteps,denois_str,hr_scale]

                    with gr.Accordion("Elemental Merge",open = False):
                        with gr.Row():
                            esettings1 = gr.CheckboxGroup(label = "settings",choices=["print change"],type="value",interactive=True)
                        with gr.Row():
                            deep = gr.Textbox(label="Blocks:Element:Ratio,Blocks:Element:Ratio,...",lines=2,value="")

                    with gr.Accordion("Tensor Merge",open = False,visible=False):
                        tensor = gr.Textbox(label="Blocks:Tensors",lines=2,value="")

                    with gr.Row():
                        x_type = gr.Dropdown(label="X type", choices=[x for x in TYPESEG], value="alpha", type="index")
                        x_randseednum = gr.Number(value=3, label="number of -1", interactive=True, visible = True)

                    #TODO: マークポイント1, XYZ対応箇所
                    xgrid = gr.Textbox(label="Sequential Merge Parameters",lines=3,value="0.25,0.5,0.75")
                    y_type = gr.Dropdown(label="Y type", choices=[y for y in TYPESEG], value="none", type="index")
                    ygrid = gr.Textbox(label="Y grid (Disabled if blank)",lines=3,value="",visible =False)
                    with gr.Row():
                        gengrid = gr.Button(elem_id="model_merger_merge", value="Sequential XY Merge and Generation",variant='primary')
                        stopgrid = gr.Button(elem_id="model_merger_merge", value="Stop XY",variant='primary')
                        s_reserve1 = gr.Button(value="Reserve XY Plot",variant='primary')
                    dtrue =  gr.Checkbox(value = True, visible = False)
                    dfalse =  gr.Checkbox(value = False,visible = False)
                    dummy_t =  gr.Textbox(value = "",visible = False)
                    #TODO: 自動マージの対応
                    #TODO: B系統のマージへの対応
                    #TODO: GPTとの統合

                blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
                with gr.Column(scale = 2):
                    sn = gr.Textbox(label="stylename", key="stylename")
                    with gr.Row():
                        # clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id="clear_prompt")
                        prompt_style_apply = uc.ToolButton(value=apply_style_symbol, elem_id="style_apply")
                        Save_style = uc.ToolButton(value=save_style_symbol, elem_id="style_create")
                    with gr.Row(elem_id="styles_row"):
                        prompt_styles = gr.Dropdown(label="Styles",interactive=True,max_choices=1, elem_id="prompt_styles", choices=[k for k, v in ps.styles.items()], value=[], multiselect=False)
                        create_refresh_button(prompt_styles, ps.reload, lambda: {"choices": [k for k, v in ps.styles.items()]}, "refresh_styles")
                        cm = gr.Textbox(label="Current Model",lines=1,value="")
                        submit_result = gr.Textbox(label="Message")
                        mgallery, mgeninfo, mhtmlinfo, mhtmllog = create_output_panel("txt2img", opts.outdir_txt2img_samples)
            with gr.Row(visible = False) as row_inputers:
                inputer = gr.Textbox(label="",lines=1,value="")
                addtox = gr.Button(value="Add to Sequence X")
                addtoy = gr.Button(value="Add to Sequence Y")
            with gr.Row(visible = False) as row_blockids:
                blockids = gr.CheckboxGroup(label = "block IDs",choices=[x for x in blockid],type="value",interactive=True)
            with gr.Row(visible = False) as row_calcmode:
                calcmodes = gr.CheckboxGroup(label = "calcmode",choices=["normal", "cosineA", "cosineB", "smoothAdd","tensor"],type="value",interactive=True)
            with gr.Row(visible = False) as row_checkpoints:
                checkpoints = gr.CheckboxGroup(label = "checkpoint",choices=[x.model_name for x in modules.sd_models.checkpoints_list.values()],type="value",interactive=True)
            with gr.Row(visible = False) as row_esets:
                esettings = gr.CheckboxGroup(label = "effective chekcer settings",choices=["save csv","save anime gif","not save grid","print change"],type="value",interactive=True)
            with gr.Tab("MBW"):
                # TODO: BASEの切り分け
                with gr.Row():
                    setMBW = gr.Button(elem_id="copy", value="set to input",variant='primary')
                    readMBW = gr.Button(elem_id="copy", value="read from weight",variant='primary')
                    setXonMBW = gr.Button(elem_id="copytogen", value="set to X",variant='primary')
                    setMBWBase = gr.Button(elem_id="copy", value="set to beta",variant='primary')
                    readMBWBase = gr.Button(elem_id="copy", value="read from beta",variant='primary')
                with gr.Row():
                    weights_mbw_base = gr.Textbox(label="weights for base(alpha and beta)",value = "0.5,0.5")
                    weights_mbw = gr.Textbox(label="weights for mbw,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
                with gr.Row():
                    with gr.Column():
                        in00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.01, value=0.5)
                        in01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.01, value=0.5)
                        in02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.01, value=0.5)
                        in03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.01, value=0.5)
                        in04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.01, value=0.5)
                        in05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.01, value=0.5)
                        in06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.01, value=0.5)
                        in07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.01, value=0.5)
                        in08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.01, value=0.5)
                        in09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.01, value=0.5)
                        in10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.01, value=0.5)
                        in11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.01, value=0.5)
                    with gr.Column():
                        base_alpha = gr.Slider(label="base_alpha", minimum=0, maximum=1, step=0.01, value=0.5)
                        base_beta = gr.Slider(label="base_beta", minimum=0, maximum=1, step=0.01, value=0.5)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                    with gr.Column():
                        ou11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.01, value=0.5)
                        ou00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.01, value=0.5)
            with gr.Tab("each MBW"):
                with gr.Row():
                    setbase = gr.Button(elem_id="copytogen", value="set to base",variant='primary')
                    readbase = gr.Button(elem_id="copytogen", value="read from base",variant='primary')
                with gr.Row():
                    with gr.Column():
                        setalpha = gr.Button(elem_id="copytogen", value="set to alpha",variant='primary')
                        readalpha = gr.Button(elem_id="copytogen", value="read from alpha",variant='primary')
                    with gr.Column():
                        setbeta = gr.Button(elem_id="copytogen", value="set to alpha",variant='primary')
                        readbeta = gr.Button(elem_id="copytogen", value="read from beta",variant='primary')
                    with gr.Row():
                        Swap = gr.Button(elem_id="copytogen", value="Swap input",variant='primary',scale=2)
                with gr.Row():
                    setalphaX = gr.Button(elem_id="copytogen", value="set alpha to X",variant='primary', scale=2)
                    setbetaX = gr.Button(elem_id="copytogen", value="set beta to X",variant='primary',scale=2)
                with gr.Row():
                    with gr.Column():
                        weights_a = gr.Textbox(label="weights for alpha, base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
                        weights_b = gr.Textbox(label="weights,for beta, base beta,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2")
                    with gr.Column():
                        weights_base = gr.Textbox(label="base,medA,medB ",value="0,0.5,0.5")
                with gr.Row():
                    with gr.Column():
                        sl_IN_A_00 = gr.Slider(label="IN_A_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_00")
                        sl_IN_A_01 = gr.Slider(label="IN_A_01", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_01")
                        sl_IN_A_02 = gr.Slider(label="IN_A_02", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_02")
                        sl_IN_A_03 = gr.Slider(label="IN_A_03", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_03")
                        sl_IN_A_04 = gr.Slider(label="IN_A_04", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_04")
                        sl_IN_A_05 = gr.Slider(label="IN_A_05", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_05")
                        sl_IN_A_06 = gr.Slider(label="IN_A_06", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_06")
                        sl_IN_A_07 = gr.Slider(label="IN_A_07", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_07")
                        sl_IN_A_08 = gr.Slider(label="IN_A_08", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_08")
                        sl_IN_A_09 = gr.Slider(label="IN_A_09", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_09")
                        sl_IN_A_10 = gr.Slider(label="IN_A_10", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_10")
                        sl_IN_A_11 = gr.Slider(label="IN_A_11", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_A_11")
                    with gr.Column():
                        sl_IN_B_00 = gr.Slider(label="IN_B_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_00")
                        sl_IN_B_01 = gr.Slider(label="IN_B_01", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_01")
                        sl_IN_B_02 = gr.Slider(label="IN_B_02", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_02")
                        sl_IN_B_03 = gr.Slider(label="IN_B_03", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_03")
                        sl_IN_B_04 = gr.Slider(label="IN_B_04", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_04")
                        sl_IN_B_05 = gr.Slider(label="IN_B_05", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_05")
                        sl_IN_B_06 = gr.Slider(label="IN_B_06", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_06")
                        sl_IN_B_07 = gr.Slider(label="IN_B_07", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_07")
                        sl_IN_B_08 = gr.Slider(label="IN_B_08", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_08")
                        sl_IN_B_09 = gr.Slider(label="IN_B_09", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_09")
                        sl_IN_B_10 = gr.Slider(label="IN_B_10", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_10")
                        sl_IN_B_11 = gr.Slider(label="IN_B_11", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_IN_B_11")
                    with gr.Column():
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        sl_Base_00 = gr.Slider(label="base", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="base")
                        sl_M_A_00 = gr.Slider(label="M_A_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_M_A_00")
                    with gr.Column():
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        sl_M_B_00 = gr.Slider(label="M_B_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_M_B_00")
                    with gr.Column():
                        sl_OUT_A_11 = gr.Slider(label="OUT_A_11", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_11")
                        sl_OUT_A_10 = gr.Slider(label="OUT_A_10", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_10")
                        sl_OUT_A_09 = gr.Slider(label="OUT_A_09", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_09")
                        sl_OUT_A_08 = gr.Slider(label="OUT_A_08", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_08")
                        sl_OUT_A_07 = gr.Slider(label="OUT_A_07", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_07")
                        sl_OUT_A_06 = gr.Slider(label="OUT_A_06", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_06")
                        sl_OUT_A_05 = gr.Slider(label="OUT_A_05", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_05")
                        sl_OUT_A_04 = gr.Slider(label="OUT_A_04", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_04")
                        sl_OUT_A_03 = gr.Slider(label="OUT_A_03", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_03")
                        sl_OUT_A_02 = gr.Slider(label="OUT_A_02", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_02")
                        sl_OUT_A_01 = gr.Slider(label="OUT_A_01", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_01")
                        sl_OUT_A_00 = gr.Slider(label="OUT_A_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_A_00")
                    with gr.Column():
                        sl_OUT_B_11 = gr.Slider(label="OUT_B_11", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_11")
                        sl_OUT_B_10 = gr.Slider(label="OUT_B_10", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_10")
                        sl_OUT_B_09 = gr.Slider(label="OUT_B_09", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_09")
                        sl_OUT_B_08 = gr.Slider(label="OUT_B_08", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_08")
                        sl_OUT_B_07 = gr.Slider(label="OUT_B_07", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_07")
                        sl_OUT_B_06 = gr.Slider(label="OUT_B_06", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_06")
                        sl_OUT_B_05 = gr.Slider(label="OUT_B_05", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_05")
                        sl_OUT_B_04 = gr.Slider(label="OUT_B_04", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_04")
                        sl_OUT_B_03 = gr.Slider(label="OUT_B_03", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_03")
                        sl_OUT_B_02 = gr.Slider(label="OUT_B_02", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_02")
                        sl_OUT_B_01 = gr.Slider(label="OUT_B_01", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_01")
                        sl_OUT_B_00 = gr.Slider(label="OUT_B_00", minimum=0, maximum=1, step=0.01, value=0.5, elem_id="sl_OUT_B_00")

            # TODO : プリセット改善
            with gr.Tab("Weights Presets"):
                with gr.Row():
                    s_reloadtext = gr.Button(value="Reload Presets",variant='primary')
                    s_reloadtags = gr.Button(value="Reload Tags",variant='primary')
                    s_savetext = gr.Button(value="Save Presets",variant='primary')
                    s_openeditor = gr.Button(value="Open TextEditor",variant='primary')
                weightstags= gr.Textbox(label="available",lines = 2,value=tagdicter(weights_presets),visible =True,interactive =True)
                wpresets= gr.TextArea(label="",value=weights_presets,visible =True,interactive  = True)

            # TODO : 多分動かないから治す
            with gr.Tab("Reservation: notWorking"):
                with gr.Row():
                    s_reserve = gr.Button(value="Reserve XY Plot",variant='primary')
                    s_reloadreserve = gr.Button(value="Reloat List",variant='primary')
                    s_startreserve = gr.Button(value="Start XY plot",variant='primary')
                    s_delreserve = gr.Button(value="Delete list(-1 for all)",variant='primary')
                    s_delnum = gr.Number(value=1, label="Delete num : ", interactive=True, visible = True,precision =0)
                with gr.Row():
                    numaframe = gr.Dataframe(
                        headers=["No.","status","xtype","xmenber", "ytype","ymenber","model A","model B","model C","alpha","beta","mode","use MBW","weights alpha","weights beta"],
                        row_count=5,)
            with gr.Row():

                currentcache = gr.Textbox(label="Current Cache")
                loadcachelist = gr.Button(elem_id="model_merger_merge", value="Reload Cache List",variant='primary')
                unloadmodel = gr.Button(value="unload model",variant='primary')


        # main ui end

        with gr.Tab("LoRA", elem_id="tab_lora"):
            pluslora.on_ui_tabs()

        with gr.Tab("History", elem_id="tab_history"):
            with gr.Row():
                with gr.Row():
                    load_history = gr.Button(value="load_history")
                with gr.Row():
                    with gr.Column():
                        history_NID = gr.Text(label="ID",lines=1,value=0)
                        def ControlIDbutton():
                            Last = uc.ToolButton(value=last_id_symbol, elem_id="LAST")
                            Prev = uc.ToolButton(value=prev_id_symbol, elem_id="PREV")
                            Next = uc.ToolButton(value=next_id_symbol,elem_id="NEXT")
                            First = uc.ToolButton(value=first_id_symbol, elem_id="FIRST")
                            HS_next = gr.Text(value="next", visible=False, elem_id="HS_NEXT")
                            HS_prev = gr.Text(value="prev", visible=False,elem_id="HS_PREV")
                            HS_last = gr.Text(value="last", visible=False,elem_id="HS_LAST")
                            HS_first = gr.Text(value="first", visible=False,elem_id="HS_FIRST")
                            HS_set = gr.Text(value="set", visible=False,elem_id="HS_SET")
                            Next.click(
                                fn=onIDchange,
                                inputs=[HS_next, history_NID],
                                outputs=[history_NID]
                            )
                            Prev.click(
                                fn=onIDchange,
                                inputs=[HS_prev, history_NID],
                                outputs=[history_NID]
                            )
                            Last.click(
                                fn=onIDchange,
                                inputs=[HS_last, history_NID],
                                outputs=[history_NID]
                            )
                            First.click(
                                fn=onIDchange,
                                inputs=[HS_first, history_NID],
                                outputs=[history_NID]
                            )
                            return Next, Prev, Last, First
                    with gr.Column():
                        ControlIDbutton()
                with gr.Row():
                        with gr.Row():
                            history_core = gr.Dataframe(max_cols=2,maxrows=2,headers=["ID", "Time", "Name"])
                        with gr.Row():
                            history_model = gr.Dataframe(headers=["Model A", "Model B", "Model C"])
                        with gr.Row():
                            history_mbw = gr.Dataframe(headers=["Weights MBW", "Base MBW"])
                        with gr.Row():
                            history_each = gr.Dataframe(headers=["Weights A", "Weights B", "Base Each"])
                            history_props = gr.Dataframe(max_cols=3,max_rows=3, headers=["Mode", "Use MBW", "Use Each", "Plus Lora", "Custom Name", "Save Setting", "Use ID"])
        with gr.Tab("Elements", elem_id="tab_deep"):
                with gr.Row():
                    smd_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint A",interactive=True)
                    create_refresh_button(smd_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                    smd_loadkeys = gr.Button(value="load keys",variant='primary')
                with gr.Row():
                    keys = gr.Dataframe(headers=["No.","block","key"],)

        #TODO : Hypernetworkのマージ対応
        #TODO : テクスチュアルのマージ対応
        #TODO : メタデータの編集対応
        #TODO : VAEのマージ対応

        with gr.Tab("Metadeta", elem_id="tab_metadata"):
                with gr.Row():
                    meta_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="read metadata",interactive=True)
                    create_refresh_button(meta_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                    smd_loadmetadata = gr.Button(value="load keys",variant='primary')
                with gr.Row():
                    metadata = gr.TextArea()


        if ub == "MBW":
            useblocks.value = True
            useeach.value = False
        elif ub == "None":
            useblocks.value = False
            useeach = False
        elif ub == "Each":
            useblocks.value = True
            useeach.value = True
        smd_loadmetadata.click(
            fn=loadmetadata,
            inputs=[meta_model_a],
            outputs=[metadata]
        )

        smd_loadkeys.click(
            fn=loadkeys,
            inputs=[smd_model_a],
            outputs=[keys]
        )


        def add_style(sn, weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, useblocks, custom_name,
                save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, useeach):
            if sn is None:
                return gr.update(value = "Please enter a name for the style")
            sp = sc.StyleDatabase(path=stylepath)
            style = sc.PromptStyle(sn,weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, useblocks, custom_name,
                save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, useeach )
            sp.styles[sn] = style
            sp.save_styles(path = stylepath)
            return gr.update(value = "Done")

        def unload():
            if shared.sd_model == None: return "already unloaded"
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)
            shared.sd_model = None
            gc.collect()
            devices.torch_gc()
            return "model unloaded"

        unloadmodel.click(fn=unload,outputs=[submit_result])

        msettings=[weights_mbw,model_a,model_b,model_c,base_alpha,base_beta,mode,calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor]
        imagegal = [mgallery,mgeninfo,mhtmlinfo,mhtmllog]
        xysettings=[x_type,xgrid,y_type,ygrid,esettings]
        MBWmenbers = [in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]
        MBWbases = [base_alpha,base_beta]
        base = [sl_Base_00,sl_M_A_00,sl_M_B_00]
        Amenbers = [sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,sl_IN_A_06,
                        sl_IN_A_07, sl_IN_A_08,sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
                        sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
                        sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11]
        Bmenbers = [sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
                        sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
                        sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
                        sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11]
        insa=[*msettings,esettings1,*xysettings,*hiresfix,batch_size,weights_a,weights_b,weights_base,ub]
        s_reverse.click(fn = reversparams,
            inputs =mergeid,
            outputs = [submit_result,*msettings[0:8],*msettings[9:13],deep,calcmode]
        )
        merge.click(
            fn=domerge,
            inputs=[*msettings,esettings1,*gensets.txt2img_preview_params, *hiresfix,batch_size, cm, dfalse,
                    in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11,
                    sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
                        sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
                        sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
                        sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
                        sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
                        sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
                        sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
                        sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11,sl_M_A_00,sl_M_B_00,sl_Base_00,radio_position_ids,txt_multi_process_cmd,useeach],
            outputs=[submit_result, cm]
        )
        mergeandgen.click(
            fn=domerge,
            inputs=[*msettings,esettings1,*gensets.txt2img_preview_params, *hiresfix,batch_size,cm,dtrue,
                    in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11,
                    sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
                        sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
                        sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
                        sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
                        sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
                        sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
                        sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
                        sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11,sl_M_A_00,sl_M_B_00,sl_Base_00,radio_position_ids,txt_multi_process_cmd,useeach],
            outputs=[submit_result, cm,*imagegal]
        )
        gen.click(
            fn=uimggen,
            inputs=[*gensets.txt2img_preview_params,*hiresfix,batch_size,cm,id_sets],
            outputs=[*imagegal],
        )

        s_reserve.click(
            fn=numaker,
            inputs=[*xysettings,*msettings,*gensets.txt2img_preview_params,*hiresfix],
            outputs=[numaframe]
        )

        s_reserve1.click(
            fn=numaker,
            inputs=[*xysettings,*msettings,*gensets.txt2img_preview_params,*hiresfix],
            outputs=[numaframe]
        )

        gengrid.click(
            fn=numanager,
            inputs=[dtrue,*xysettings,*msettings,*gensets.txt2img_preview_params,*hiresfix],
            outputs=[submit_result,cm,*imagegal],
        )

        s_startreserve.click(
            fn=numanager,
            inputs=[dfalse,*xysettings,*msettings,*gensets.txt2img_preview_params,*hiresfix],
            outputs=[submit_result,cm,*imagegal],
        )

        load_history.click(fn = load_historyf,inputs=[history_NID],outputs=[history_core, history_model, history_mbw, history_each, history_props])

        s_reloadreserve.click(fn=nulister,inputs=[dfalse],outputs=[numaframe])
        s_delreserve.click(fn=nulister,inputs=[s_delnum],outputs=[numaframe])
        loadcachelist.click(fn=load_cachelist,inputs=[],outputs=[currentcache])
        addtox.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[inputer],outputs=[xgrid])
        addtoy.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[inputer],outputs=[ygrid])

        stopgrid.click(fn=freezetime)
        stopmerge.click(fn=freezemtime)

        checkpoints.change(fn=lambda x:",".join(x),inputs=[checkpoints],outputs=[inputer])
        blockids.change(fn=lambda x:" ".join(x),inputs=[blockids],outputs=[inputer])
        calcmodes.change(fn=lambda x:",".join(x),inputs=[calcmodes],outputs=[inputer])

        setbase.click(fn=text2base,inputs=weights_base,outputs=[*base])
        setalpha.click(fn=slider2mod,inputs=Amenbers,outputs=[weights_a])
        setbeta.click(fn=slider2mod,inputs=Bmenbers,outputs=[weights_b])
        setMBW.click(fn=slider2mod,inputs=MBWmenbers,outputs=[weights_mbw_base])
        setMBWBase.click(fn=slider2mod,inputs=MBWbases,outputs=[weights_mbw])

        setXonMBW.click(fn=add_to_seq,inputs=[xgrid,weights_mbw],outputs=[xgrid])
        setalphaX.click(fn=add_to_seq,inputs=[xgrid,weights_a],outputs=[xgrid])
        setbetaX.click(fn=add_to_seq,inputs=[xgrid,weights_b],outputs=[xgrid])
        readbase.click(fn=base2text,inputs=base,outputs=weights_base)
        readMBWBase.click(fn=text2slider,inputs=weights_mbw,outputs=[*MBWbases])
        readMBW.click(fn=text2slider,inputs=weights_mbw_base,outputs=[*MBWmenbers])
        readalpha.click(fn=text2mod,inputs=weights_a,outputs=[*Amenbers])
        readbeta.click(fn=text2mod,inputs=weights_b,outputs=[*Bmenbers])
        Swap.click(fn=text2slider,inputs=[weights_a,weights_b],outputs=[weights_a,weights_b,*Amenbers,*Bmenbers])

        x_type.change(fn=showxy,inputs=[x_type,y_type], outputs=[row_blockids,row_checkpoints,row_inputers,ygrid,row_esets,row_calcmode])
        y_type.change(fn=showxy,inputs=[x_type,y_type], outputs=[row_blockids,row_checkpoints,row_inputers,ygrid,row_esets,row_calcmode])
        x_randseednum.change(fn=makerand,inputs=[x_randseednum],outputs=[xgrid])
        Save_style.click(
                fn=add_style,
                # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                # the same number of parameters, but we only know the style-name after the JavaScript prompt
                inputs=[sn,*insa],
                outputs=[submit_result],
            )
        prompt_style_apply.click(
                fn=ps.get_style_prompts,
                inputs=[prompt_styles],
                outputs=[weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep,
                        tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson,
                        hrupscaler, hr2ndsteps, denois_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub])
        import subprocess
        def openeditors():
            subprocess.Popen(['start', filepath], shell=True)

        def reloadpresets():
            try:
                with open(filepath) as f:
                    return f.read()
            except OSError as e:
                pass
        def savepresets(text):
            with open(filepath,mode = 'w') as f:
                f.write(text)

        s_reloadtext.click(fn=reloadpresets,inputs=[],outputs=[wpresets])
        s_reloadtags.click(fn=tagdicter,inputs=[wpresets],outputs=[weightstags])
        s_savetext.click(fn=savepresets,inputs=[wpresets],outputs=[])
        s_openeditor.click(fn=openeditors,inputs=[],outputs=[])
        return [(ui, "UltimetMerger", "UltimetMerger")]


msearch = []
mlist=[]

def loadmetadata(model):
    import json
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    if ".safetensors" not in checkpoint_info.filename: return "no metadata(not safetensors)"
    sdict = sd_models.read_metadata_from_safetensors(checkpoint_info.filename)
    if sdict == {}: return "no metadata"
    return json.dumps(sdict,indent=4)

def searchhistory(words,searchmode):
    outs =[]
    ando = "and" in searchmode
    words = words.split(" ") if " " in words else [words]
    for i, m in  enumerate(msearch):
        hit = ando
        for w in words:
            if ando:
                if w not in m:hit = False
            else:
                if w in m:hit = True
        print(i,len(mlist))
        if hit :outs.append(mlist[i])

    if outs == []:return [["no result","",""],]
    return outs

# TODO: マークポイント2, XYZの対応箇所(関数) + 別のpyファイル化
def reversparams(id):
    def selectfromhash(hash):
        for model in sd_models.checkpoint_tiles():
            if hash in model:
                return model
        return ""
    try:
        idsets = rwmergelog(id = id)
    except:
        return [gr.update(value = "ERROR: history file could not open"),*[gr.update() for x in range(14)]]
    if type(idsets) == str:
        print("ERROR")
        return [gr.update(value = idsets),*[gr.update() for x in range(14)]]
    if idsets[0] == "ID":return  [gr.update(value ="ERROR: no history"),*[gr.update() for x in range(14)]]
    mgs = idsets[3:]
    if mgs[0] == "":mgs[0] = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    if mgs[1] == "":mgs[1] = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
    mgs[2] = selectfromhash(mgs[2]) if len(mgs[2]) > 5 else ""
    mgs[3] = selectfromhash(mgs[3]) if len(mgs[3]) > 5 else ""
    mgs[4] = selectfromhash(mgs[4]) if len(mgs[4]) > 5 else ""
    mgs[8] = True if mgs[8] =="True" else False
    mgs[10] = mgs[10].replace("[","").replace("]","").replace("'", "")
    mgs[10] = [x.strip() for x in mgs[10].split(",")]
    mgs[11] = mgs[11].replace("[","").replace("]","").replace("'", "")
    mgs[11] = [x.strip() for x in mgs[11].split(",")]
    while len(mgs) < 14:
        mgs.append("")
    mgs[13] = "normal" if mgs[13] == "" else mgs[13]
    return [gr.update(value = "setting loaded") ,*[gr.update(value = x) for x in mgs[0:14]]]

def add_to_seq(seq,maker):
    return gr.Textbox.update(value = maker if seq=="" else seq+"\r\n"+maker)

def load_cachelist():
    text = ""
    for x in checkpoints_loaded.keys():
        text = text +"\r\n"+ x.model_name
    return text.replace("\r\n","",1)

def makerand(num):
    text = ""
    for x in range(int(num)):
        text = text +"-1,"
    text = text[:-1]
    return text

#row_blockids,row_checkpoints,row_inputers,ygrid
def showxy(x,y):
    flags =[False]*6
    t = TYPESEG
    txy = t[x] + t[y]
    if "model" in txy : flags[1] = flags[2] = True
    if "pinpoint" in txy : flags[0] = flags[2] = True
    if "effective" in txy or "element" in txy : flags[4] = True
    if "calcmode" in txy : flags[5] = True
    if not "none" in t[y] : flags[3] = flags[2] = True
    return [gr.update(visible = x) for x in flags]

def text2slider(text):
    vals = [t.strip() for t in text.split(",")]
    return [gr.update(value = float(v)) for v in vals]


def text2mod(text):
    vals = [t.strip() for t in text.split(",")]
    return [gr.update(value = float(v)) for v in vals]

def slider2mod(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x):
    numbers = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x]
    numbers = [str(x) for x in numbers]
    return gr.update(value = ",".join(numbers) )




def text2base(text):
    vals = [t.strip() for t in text.split(",")]
    return [gr.update(value = float(v)) for v in vals]

def base2text(a,c,d):
    numbers = [a,c,d]
    numbers = [str(x) for x in numbers]
    return gr.update(value = ",".join(numbers) )


def swap(weights_a, weights_b):
    i = weights_a
    x = weights_b
    weights_a = x
    weights_b = i
    b = text2slider(weights_a)
    a = text2slider(weights_b)
    return gr.update(value=weights_a), gr.update(value=weights_b), a, b
def tagdicter(presets):
    presets=presets.splitlines()
    wdict={}
    for l in presets:
        w=[]
        if ":" in l :
            key = l.split(":",1)[0]
            w = l.split(":",1)[1]
        if "\t" in l:
            key = l.split("\t",1)[0]
            w = l.split("\t",1)[1]
        if len([w for w in w.split(",")]) == 26:
            wdict[key.strip()]=w
    return ",".join(list(wdict.keys()))

def loadkeys(model_a):
    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    sd = sd_models.read_state_dict(checkpoint_info.filename,"cpu")
    keys = []
    for i, key in enumerate(sd.keys()):
        re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
        re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
        re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

        weight_index = -1
        blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11","Not Merge"]

        NUM_INPUT_BLOCKS = 12
        NUM_MID_BLOCK = 1
        NUM_OUTPUT_BLOCKS = 12
        NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

        if 'time_embed' in key:
            weight_index = -2                # before input blocks
        elif '.out.' in key:
            weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
        else:
            m = re_inp.search(key)
            if m:
                inp_idx = int(m.groups()[0])
                weight_index = inp_idx
            else:
                m = re_mid.search(key)
                if m:
                    weight_index = NUM_INPUT_BLOCKS
                else:
                    m = re_out.search(key)
                    if m:
                        out_idx = int(m.groups()[0])
                        weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx
        keys.append([i,blockid[weight_index+1],key])
    return keys

def block_merger(in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11
    ):

        _weight_A = ",".join(
            [str(x) for x in [
                in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11
            ]])
        _weight_B = ",".join(
            [str(x) for x in [
                in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11
            ]])

        return _weight_A, _weight_B
def each_merger(model_a,model_b,
    sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
    sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
    sl_M_A_00,
    sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
    sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
    sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
    sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
    sl_M_B_00,
    sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
    sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11, sl_Base_00,submit_result
    ):

        _weight_A = ",".join(
            [str(x) for x in [
                sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
                sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
                sl_M_A_00,
                sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
                sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
            ]])
        _weight_B = ",".join(
            [str(x) for x in [
                sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
                sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
                sl_M_B_00,
                sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
                sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11,
            ]])

        # debug output
        print( "#### Merge Block Weighted : Each ####")

        if (not model_a or not model_b) and submit_result == "":
            _err_msg = f"ERROR: model not found. [{model_a}][{model_b}]"
            print(_err_msg)
            return gr.update(value=_err_msg)

        ret_html = ""
        if submit_result != "":
            # need multi-merge
            _lines = submit_result.split('\n')
            print(f"check multi-merge. {len(_lines)} lines found.")
            for line_index, _line in enumerate(_lines):
                if _line == "":
                    continue
                print(f"\n== merge line {line_index+1}/{len(_lines)} ==")
                _items = [x.strip() for x in _line.split(",") if x != ""]
                if len(_items) > 0:
                    return _weight_A, _weight_B
        else:
            # normal merge
            ret_html

        sd_models.list_models()
        print( "#### All merge process done. ####")

        return gr.update(value=f"{ret_html}")
def apply_styles(prompt_styles,Amenbers,Bmenbers,Base):
    ps = sc.StyleDatabase(path = stylepath)
    weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep,tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson,hrupscaler, hr2ndsteps, denois_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub= ps.get_style_prompts(prompt_styles)
    Base = text2base(weights_base)
    Amenbers = text2slider(weights_a)
    Bmenbers = text2slider(weights_b)
    list = []
    for i in Base:
        list.append(gr.update(value=i))
    for i in Amenbers:
        list.append(gr.update(value=i))
    for i in Bmenbers:
        list.append(gr.update(value=i))
    return weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep,tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson,hrupscaler, hr2ndsteps, denois_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub
def domerge(weights_mbw,model_a,model_b,model_c,base_alpha,base_beta,mode,
calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,
esettings,
prompt,nprompt,steps,sampler,cfg,seed,w,h,
hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,
currentmodel,imggen,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11,
sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11, sl_M_A_00,sl_M_B_00,sl_Base_00, radio_position_ids,submit_result,useeach):
    if useeach == False:
        print("Skip Each mode")
        weights_a,weights_b = block_merger(in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11)
        img = None
        if imggen == False:
            return umergegen(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,
            calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,
            esettings,
            prompt,nprompt,steps,sampler,cfg,seed,w,h,
            hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,
            currentmodel,imggen)
        elif imggen == True:
            return umergegen(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,
            calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,
            esettings,
            prompt,nprompt,steps,sampler,cfg,seed,w,h,
            hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,
            currentmodel,imggen)
    else:
        wA, wB = each_merger(model_a,model_b,
        sl_IN_A_00, sl_IN_A_01, sl_IN_A_02, sl_IN_A_03, sl_IN_A_04, sl_IN_A_05,
        sl_IN_A_06, sl_IN_A_07, sl_IN_A_08, sl_IN_A_09, sl_IN_A_10, sl_IN_A_11,
        sl_M_A_00,
        sl_OUT_A_00, sl_OUT_A_01, sl_OUT_A_02, sl_OUT_A_03, sl_OUT_A_04, sl_OUT_A_05,
        sl_OUT_A_06, sl_OUT_A_07, sl_OUT_A_08, sl_OUT_A_09, sl_OUT_A_10, sl_OUT_A_11,
        sl_IN_B_00, sl_IN_B_01, sl_IN_B_02, sl_IN_B_03, sl_IN_B_04, sl_IN_B_05,
        sl_IN_B_06, sl_IN_B_07, sl_IN_B_08, sl_IN_B_09, sl_IN_B_10, sl_IN_B_11,
        sl_M_B_00,
        sl_OUT_B_00, sl_OUT_B_01, sl_OUT_B_02, sl_OUT_B_03, sl_OUT_B_04, sl_OUT_B_05,
        sl_OUT_B_06, sl_OUT_B_07, sl_OUT_B_08, sl_OUT_B_09, sl_OUT_B_10, sl_OUT_B_11, base_alpha,submit_result)
        base_alpha = sl_Base_00
        base_beta = sl_Base_00
        print("start Each")
        img = ""
        if imggen == False:
            return eachmergegen(wA,wB,model_a,model_b,model_c,base_alpha,base_beta,mode,
                calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,
                esettings,
                prompt,nprompt,steps,sampler,cfg,seed,w,h,
                hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,
                currentmodel,radio_position_ids,imggen)
        elif imggen == True:
            return eachmergegen(wA,wB,model_a,model_b,model_c,base_alpha,base_beta,mode,
                calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,
                esettings,
                prompt,nprompt,steps,sampler,cfg,seed,w,h,
                hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,
                currentmodel,radio_position_ids,imggen)



script_callbacks.on_ui_train_tabs(on_ui_train_tabs)
script_callbacks.on_ui_tabs(on_ui_tabs)
