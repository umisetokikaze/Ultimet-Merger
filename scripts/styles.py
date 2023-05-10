import csv
import os
import shutil
import tempfile
import typing

import gradio as gr
import modules.ui_components as uc

system_paste_fields = []
refresh_symbol = '\U0001f504'  # 🔄
paste_fields = {}
registered_param_bindings = []
styledir=  "./umerge_style"
stylepath = "./umerge_style/style.csv"
class ParamBinding:
    def __init__(self, paste_button, tabname, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=[]):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names
class PromptStyle(typing.NamedTuple):
    name:str
    weights_mbw:str
    model_a:str
    model_b:str
    model_c:str
    base_alpha:str
    base_beta:str
    mode:str
    calcmode:str
    useblocks:str
    custom_name:str
    save_sets:str
    id_sets:str
    wpresets:str
    deep:str
    tensor:str
    esettings1:str
    x_type:str
    xgrid:str
    y_type:str
    ygrid:str
    esettings:str
    hireson:str
    hrupscaler:str
    hr2ndsteps:str
    denoise_str:str
    hr_scale:str
    batch_size:str
    weights_a:str
    weights_b:str
    weights_base:str
    ub:str
# def merge_prompts(style_prompt: str, prompt: str) -> str:
#     if "{prompt}" in style_prompt:
#         res = style_prompt.replace("{prompt}", prompt)
#     else:
#         parts = filter(None, (prompt.strip(), style_prompt.strip()))
#         res = ", ".join(parts)

#     return res


# def apply(prompt, styles):
#     for style in styles:
#         prompt = merge_prompts(style, prompt)

#     return prompt


class StyleDatabase:
    def __init__(self, path: str):
        self.no_style = PromptStyle("None","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",)
        self.styles = {}
        self.path = stylepath

        self.reload()

    def reload(self):
        self.styles.clear()

        if not os.path.exists(self.path):
            print(f'Creating styles database: {self.path}')
            self.save_styles(self.path)

        with open(self.path, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                name = row["name"] if "name" in row else row["text"]
                weights_mbw= row.get("weights_mbw", "")
                model_a= row.get("model_a", "")
                model_b= row.get("model_b", "")
                model_c= row.get("model_c", "")
                base_alpha= row.get("base_alpha", 0.5)
                base_beta= row.get("base_beta", 0.25)
                mode= row.get("mode",  "Weight sum:A*(1-alpha)+B*alpha")
                calcmode= row.get("calcmode", "None")
                useblocks= row.get("useblocks", "")
                custom_name= row.get("custom_name", "")
                save_sets= row.get("save_sets", "")
                id_sets= row.get("id_sets", "")
                wpresets= row.get("wpresets", "")
                deep= row.get("deep", "")
                tensor= row.get("tensor", "")
                esettings1= row.get("esettings1", "")
                x_type= row.get("x_type", "alpha")
                xgrid= row.get("xgrid", "")
                y_type= row.get("y_type", "none")
                ygrid= row.get("ygrid", "")
                esettings= row.get("esettings", "")
                hireson= row.get("hireson", False)
                hrupscaler= row.get("hrupscaler", "")
                hr2ndsteps= row.get("hr2ndsteps", 0)
                denoise_str= row.get("denoise_str", 0.7)
                hr_scale= row.get("hr_scale", 2)
                batch_size= row.get("batch_size", 1)
                weights_a= row.get("weights_a", "")
                weights_b= row.get("weights_b", "")
                weights_base= row.get("weights_base", "0")
                ub= row.get("ub", "")
                self.styles[row["name"]] = PromptStyle(row["name"], weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, useblocks, custom_name,
                save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub)

    def get_style_prompts(self,name):
        if name == "None":
            return []
        items = []
        out = []
        with open(self.path, "r", encoding="utf-8-sig", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Support loading old CSV format with "name, text"-columns
                    name = row["name"] if "name" in row else row["text"]
                    weights_mbw= row.get("weights_mbw", "")
                    model_a= row.get("model_a", "")
                    model_b= row.get("model_b", "")
                    model_c= row.get("model_c", "")
                    base_alpha= row.get("base_alpha", 0.5)
                    base_beta= row.get("base_beta", 0.25)
                    mode= row.get("mode",  "Weight sum:A*(1-alpha)+B*alpha")
                    calcmode= row.get("calcmode", "None")
                    custom_name= row.get("custom_name", "")
                    save_sets= row.get("save_sets", "")
                    id_sets= row.get("id_sets", "")
                    wpresets= row.get("wpresets", "")
                    deep= row.get("deep", "")
                    tensor= row.get("tensor", "")
                    esettings1= row.get("esettings1", "")
                    x_type= row.get("x_type", "alpha")
                    xgrid= row.get("xgrid", "")
                    y_type= row.get("y_type", "none")
                    ygrid= row.get("ygrid", "")
                    esettings= row.get("esettings", "")
                    hireson= row.get("hireson", False)
                    hrupscaler= row.get("hrupscaler", "")
                    hr2ndsteps= row.get("hr2ndsteps", 0)
                    denoise_str= row.get("denoise_str", 0.7)
                    hr_scale= row.get("hr_scale", 2)
                    batch_size= row.get("batch_size", 1)
                    weights_a= row.get("weights_a", "")
                    weights_b= row.get("weights_b", "")
                    weights_base= row.get("weights_base", "0")
                    ub= row.get("ub", "")
                    if row["name"] == name:
                        items = weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub
        for item in items:
            out.append(gr.update(visible=True, value=item))
        weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub = out
        return weights_mbw, model_a, model_b, model_c, base_alpha, base_beta, mode, calcmode, custom_name,save_sets, id_sets, wpresets, deep, tensor, esettings1, x_type, xgrid, y_type, ygrid, esettings, hireson, hrupscaler, hr2ndsteps, denoise_str, hr_scale, batch_size, weights_a, weights_b, weights_base, ub

    # def apply_styles(self, prompt, styles):
    #     return [self.styles.get(x, self.no_style).prompt for x in styles]


    def save_styles(self,path: str) -> None:
        # Write to temporary file first, so we don't nuke the file if something goes wrong
        fd, temp_path = tempfile.mkstemp(".csv")
        if not os.path.exists(styledir):
            os.makedirs(styledir)
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
            # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
            writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
            writer.writeheader()
            writer.writerows(style._asdict() for k, style in self.styles.items())
        shutil.move(temp_path, path)
        return "done"
