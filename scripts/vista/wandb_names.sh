# W&B display names, sourced by sbatch_repro_finetune.sh and sbatch_eval_run.sh.
# Format: <start>_<training>_<task mix>. "pretrained" starts from the 200k parent; "ft-Nh" is fine-tuning
# on N hours of task data; "scratch-*" models are trained from new weights ("train-Nh"); "multi6" counts
# all 6 tasks' hours (12 h of it put_relative). The run dir stays the W&B run ID, so links and resumes
# are unaffected; renamed 2026-11-02. Unlisted runs keep their run-dir name.
wandb_display_name() {
  case "$1" in
    repro_sim_put_the_plastic_bottles_in_the_bin_recipe) echo pretrained_ft-49h_single_bottles ;;
    repro_put_relative_frac100)          echo pretrained_ft-12h_single ;;
    repro_put_relative_frac50)           echo pretrained_ft-6h_single ;;
    repro_put_relative_frac25)           echo pretrained_ft-3h_single ;;
    repro_put_relative_frac10)           echo pretrained_ft-1.2h_single ;;
    repro_mt6_pretrained)                echo pretrained_ft-87h_multi6 ;;
    repro_put_relative_scratch_dino)     echo scratch-dinov3_train-12h_single ;;
    repro_put_relative_scratch_rand)     echo scratch-random_train-12h_single ;;
    repro_mt6_scratch_dino)              echo scratch-dinov3_train-87h_multi6 ;;
    repro_mt6_scratch_rand)              echo scratch-random_train-87h_multi6 ;;
    repro_put_relative_scratch_dino_75k) echo scratch-dinov3_train-12h_single_75k ;;
    repro_put_relative_ft_lr1e5)         echo pretrained_ft-1.2h_single_lr1e-5 ;;
    repro_put_relative_ft_vis01)         echo pretrained_ft-1.2h_single_vision-lr0.1 ;;
    put_relative_parent)                 echo pretrained_no-ft ;;
    put_relative_released)               echo released_abc-ft_single ;;
    *)                                   echo "$1" ;;
  esac
}
