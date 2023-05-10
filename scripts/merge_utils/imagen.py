from modules import shared, processing, sd_models, images, sd_samplers,scripts
from modules import shared, processing, sd_models, images, sd_samplers,scripts
from modules.ui import  plaintext_to_html
from scripts.merge_utils.utils import draw_origin
from modules.processing import create_infotext,Processed
from modules.shared import opts

def uimggen(prompt, nprompt, steps, sampler, cfg, seed, w, h,hireson,hrupscaler,hr2ndsteps,denoise_str,hr_scale,batch_size,mergeinfo="",id_sets=[],modelid = "no id"):
    shared.state.begin()
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        do_not_save_grid=True,
        do_not_save_samples=True,
        do_not_reload_embeddings=True,
    )
    p.batch_size = int(batch_size)
    p.prompt = prompt
    p.negative_prompt = nprompt
    p.steps = steps
    p.sampler_name = sd_samplers.samplers[sampler].name
    p.cfg_scale = cfg
    p.seed = seed
    p.width = w
    p.height = h
    p.seed_resize_from_w=0
    p.seed_resize_from_h=0
    p.denoising_strength=None
    if hireson:
        p.enable_hr = hireson
        p.denoising_strength = denoise_str
        p.hr_upscaler = hrupscaler
        p.hr_second_pass_steps = hr2ndsteps
        p.hr_scale = hr_scale

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    processed:Processed = processing.process_images(p)
    if "image" in id_sets:
        for i, image in enumerate(processed.images):
            processed.images[i] = draw_origin(image, str(modelid),w,h,w)

    if "PNG info" in id_sets:mergeinfo = mergeinfo + " ID " + str(modelid)

    infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)
    if infotext.count("Steps: ")>1:
        infotext = infotext[:infotext.rindex("Steps")]

    infotexts = infotext.split(",")
    for i,x in enumerate(infotexts):
        if "Model:"in x:
            infotexts[i] = " Model: "+mergeinfo.replace(","," ")
    infotext= ",".join(infotexts)

    for i, image in enumerate(processed.images):
        images.save_image(image, opts.outdir_txt2img_samples, "",p.seed, p.prompt,shared.opts.samples_format, p=p,info=infotext)

    if batch_size > 1:
        grid = images.image_grid(processed.images, p.batch_size)
        processed.images.insert(0, grid)
        images.save_image(grid, opts.outdir_txt2img_grids, "grid", p.seed, p.prompt, opts.grid_format, info=infotext, short_filename=not opts.grid_extended_filename, p=p, grid=True)
    shared.state.end()
    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments),p
