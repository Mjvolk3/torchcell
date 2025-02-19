#!/bin/bash

wandb_sync_agent_dirs() {
    # Exit on error, undefined vars, and pipe failures
    set -euo pipefail

    # Check if arguments were provided
    if [ $# -eq 0 ]; then
        echo "Error: No directories provided"
        echo "Usage: wandb_sync_agent_dirs dir1 dir2 ..."
        return 1
    fi  # Changed from } to fi

    # Source conda functions for non-interactive shells
    CONDA_BASE=$(conda info --base)
    if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        echo "Error: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
        return 1
    fi  # Changed from } to fi

    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    # Activate conda environment safely
    if ! conda activate torchcell 2>/dev/null; then
        echo "Error: Failed to activate torchcell environment"
        return 1
    fi  # Changed from } to fi

    # Export ADDR2LINE to prevent unbound variable warning
    export ADDR2LINE=""

    # Process each directory
    for exp_dir in "$@"; do
        wandb_dir="${exp_dir}/wandb"
        if [[ -d "${wandb_dir}" ]]; then
            echo "Syncing runs in ${wandb_dir}..."
            (
                cd "${wandb_dir}" || exit 1
                for d in offline-run-*/; do
                    if [[ -d "$d" ]]; then
                        echo "Syncing ${d}..."
                        wandb sync "${d}"
                    fi
                done
            )
        else
            echo "Directory ${wandb_dir} not found, skipping..."
        fi
    done
}

# Make the function available for export
export -f wandb_sync_agent_dirs

# wandb_sync_agent_dirs \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535046_bf58cfe7e8d8117382e460b3af284ee441c8764da79870420147194156d98c28" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535043_d030fb268eda9c675d05f82057f09f3bf35162e85406b541dce6c4e2cc612a1c" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535042_dc5ed3baae554400cf6d7967184fd1c4db70ecd7388e0e2ce221769bc5b8cc5f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535041_41e4a1765b750b7d71e74e5d88db88d156b3225313296cd488cb076ecb571140" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535040_9a673275ed83899c72946791093b79a20f398bfa86858597d0e087d13f854909"

# Spaces necessary


# wandb_sync_agent_dirs \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536312_d64fbe010531564be53897fea8b11fa49508a0fab254371889eb2205c720e1b4" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536312_66faf8a5ebd6c9e4e4f05255332b7555b14a134a6aa196c829f9e5fed023d253" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536312_7c19b9549c39993c6529c0e6ebad1cdf4911910a13960d423f88788f2b6a1ece" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536312_0c4c8e9106c03e10e7255c287b479d43b4bed7c9cbb65bbf00337183601559bc" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536309_d5817459ac9ca6ff219488958f6a4c3cbbe8181cf4129015133e8aca922f0f25" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536301_ed775c8c5ae5c80a6614417541fda1016f5479c7e9a043b254368c974af8163e" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536301_df6645086e0b012ba326352824569c45e28295754752c2fe59ebd2ee98e791a7" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536301_c3dbdee5705c2ae3e16f238a470e988df87d9d9786faf9696a33cfea522cce32" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536301_a418f91816f6dcfb7a879738dc46af35829eef61c5e943b897f722994a3d3b1f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536293_e6915cdb19b06cb745218980c08e082defaf59ee7842a082da22c0fee59aca05" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536293_b51a86bb56d18ec485f97000adcd04e7bcea0b9ef61d0aab42c2c75f86d00abf" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536293_b24c8b910e323f6eb3d5ecca1ecd833945190090432a186469d3dd5654fa241f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536293_31bf225589446eb6f2deeeef7d27e8a46d6f7487ca5113dad57ad266541108cf" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536287_ee9a90e22006ce4cceb08158fc49680ae485fe0d2e838c05f2865cc706ba1236" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536287_b1c5184ed37c026b0ec3f93ce6cf37fb94987200f12867f3f0bb9a49d1596c46" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536287_3eb3ebb59957342dfdd68a27db7812a2f8cafdec841336baa6acf024d93c0b01" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536282_ee9a90e22006ce4cceb08158fc49680ae485fe0d2e838c05f2865cc706ba1236" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536282_b1c5184ed37c026b0ec3f93ce6cf37fb94987200f12867f3f0bb9a49d1596c46" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536277_ee9a90e22006ce4cceb08158fc49680ae485fe0d2e838c05f2865cc706ba1236" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536277_b1c5184ed37c026b0ec3f93ce6cf37fb94987200f12867f3f0bb9a49d1596c46" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536272_ee9a90e22006ce4cceb08158fc49680ae485fe0d2e838c05f2865cc706ba1236" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536272_b1c5184ed37c026b0ec3f93ce6cf37fb94987200f12867f3f0bb9a49d1596c46" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536272_3eb3ebb59957342dfdd68a27db7812a2f8cafdec841336baa6acf024d93c0b01" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536264_fe7517fa67d699da1fdc7d0521052b83d56ffda5c4f6aa2dd06cfcf04e8c1cf9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536264_c43e01c35c50634563c5844f50454d1bde0540b3798d24ee090ecc3b62e9ca7f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536264_1924760e9013fdc34f217f0fce9267bfcfa2b538ba40acdb75e48f5eb7dc38a3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536264_538d7fa99856d4a485de7316d29562ff48b6b51b898ae1245d4a3365aaa3406d" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_e7b4156819342e50eae029b317399da20f1ccd10f50af4db9d0d241f81db7da9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_caa7229419bc8f77d5b6ae4e35e760856659e2158513b2c9bf46e614fdac6062" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_b9ca0ec96b7a0a3a46d54bccdaa0e4accad0eb2d74ce10622a3383e1912094dc" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_b1c5184ed37c026b0ec3f93ce6cf37fb94987200f12867f3f0bb9a49d1596c46" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_478d94e4f0cc782ef64ba94399728551f6079a59b889a9ef2e30fd7f615e7673" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_30b21725987143feb8fd5ba9573c42313868ccfc2257c224525795a150e6c08e" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536220_11bba276de6096510da18ed7a79d3b4ac47f60a405a2c7ab9e7fb9f055bed52e" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536214_b9ca0ec96b7a0a3a46d54bccdaa0e4accad0eb2d74ce10622a3383e1912094dc" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536214_acffc61c54b66bf9f7d5ae297028dfa5f5874a0e233341d8d541c6bad9f520e2" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536214_3286af6093e65368ba45633e94f5fbcdec3484d048de1eca0d16c03bfc808e65" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536186_acffc61c54b66bf9f7d5ae297028dfa5f5874a0e233341d8d541c6bad9f520e2" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536186_3286af6093e65368ba45633e94f5fbcdec3484d048de1eca0d16c03bfc808e65" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536141_e8e2a18a9b4524a40c13f2c58e5bcb48c1a7ba29d50508b30261617b9f7204fe" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536141_7f694cc7f516d24fdd73e07534facd150894bb79c66d4aea3c0537b7277f1fca" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536133_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536133_e8e2a18a9b4524a40c13f2c58e5bcb48c1a7ba29d50508b30261617b9f7204fe" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536133_7f694cc7f516d24fdd73e07534facd150894bb79c66d4aea3c0537b7277f1fca" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536131_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536130_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536129_e8e2a18a9b4524a40c13f2c58e5bcb48c1a7ba29d50508b30261617b9f7204fe" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536128_7f694cc7f516d24fdd73e07534facd150894bb79c66d4aea3c0537b7277f1fca" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536123_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536122_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536121_e8e2a18a9b4524a40c13f2c58e5bcb48c1a7ba29d50508b30261617b9f7204fe" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536118_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536117_f08f228b6a84c40fe5272ee9be5976a9cfec2f460dbc743e167f0fd47336f6f3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1536116_e8e2a18a9b4524a40c13f2c58e5bcb48c1a7ba29d50508b30261617b9f7204fe" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535046_bf58cfe7e8d8117382e460b3af284ee441c8764da79870420147194156d98c28" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535043_d030fb268eda9c675d05f82057f09f3bf35162e85406b541dce6c4e2cc612a1c" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535042_dc5ed3baae554400cf6d7967184fd1c4db70ecd7388e0e2ce221769bc5b8cc5f" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535041_41e4a1765b750b7d71e74e5d88db88d156b3225313296cd488cb076ecb571140" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535040_9a673275ed83899c72946791093b79a20f398bfa86858597d0e087d13f854909" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535039_f74a9a66b40e5af111fb8483126d5fbc5fe148576f56343279ca57a9943343df" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535037_6c2eee0204f33046bffcd72628b365d2d35208e7a4b762642e59a9fcae1907a3" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535036_fb96f4af74796c807669eb21beca35d0cce0727dfe8301c97a4bd9a5546d47a8" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535036_f49f0fca32ea126a36cad31827b0b65525884768a036c51ad9d0db8092edc265" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535036_ee4c8e3ccbda832b4e9ff9f1a376b7301497708ef0e02fe4a1e9d49c108bfded" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535036_d644372853c11518af196943937716d5365dd29ac974ade77b2eeae9c63f3fd9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535035_fb96f4af74796c807669eb21beca35d0cce0727dfe8301c97a4bd9a5546d47a8" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535035_f49f0fca32ea126a36cad31827b0b65525884768a036c51ad9d0db8092edc265" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535035_ee4c8e3ccbda832b4e9ff9f1a376b7301497708ef0e02fe4a1e9d49c108bfded" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535035_d644372853c11518af196943937716d5365dd29ac974ade77b2eeae9c63f3fd9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535032_fd48e976d82aa3f61c42cf0f0ff8ef69317631d00112b8d8d9c61d46e12ec2d6" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535032_f49f0fca32ea126a36cad31827b0b65525884768a036c51ad9d0db8092edc265" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535032_ee4c8e3ccbda832b4e9ff9f1a376b7301497708ef0e02fe4a1e9d49c108bfded" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535032_d644372853c11518af196943937716d5365dd29ac974ade77b2eeae9c63f3fd9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535029_f71fd94e52be005596dda810306638fa24b8111018006bf90ae1484768c61929" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535029_bb8baf90e26aaba1f4af8e0eb085af6b613f89460bfb8319392742dc160c30e9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535029_2652bbe3a946e136f1ee5c530aaeb6ef0febbd54bb97e15d8632952c1c6c09b5" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535029_67aad5660fc00956a12200323c6c32c54912c4925fadfe55339c07aaf56648ef" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535028_ae5b9635fd9825e30cc728a009cdb853359a6989b3459f061c7c968ab523a168" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535028_887c83a219e3003f2e08f35f9fedcb7d18246b5baa79747a60886d504472a74e" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535028_96b61c46a567ead96e3022b52979564984ad55aedb0f4f0e20e9adc5128d78e2" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535028_7d8774cf0862b5b93234136efd1ced49edc8ea724713a816d8d704fe24aa04d6" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535026_033504665ea94351f3da24ccd331654c2f4acbef429504ef59528e756de5e309" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535026_8798eabbf6d775618a87a472eca2a5bf3bdc4435d747ad5af1914110fdb768b8" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535026_600ba684bb02da5301f91751b730345df0bdaeff86b871bc76ef0b396f7a424e" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535026_532fadf44eb36cbb8890c356273824f0208287d08bdb5b8683a407c06a1f5296" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535024_f3f74a694b5e80a852315d726ab1a39e4b9be88fc36b7149947cf2c688921cc7" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535023_f1d1124e09143ba1705ca527aaf1d05789301ab7022295d4115fa0749d96fa58" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535023_e3be27fefe00bafb6388c5113c0cb60d17ce24737ed6903274063387be2faab0" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535023_8589566a53a4148624227a6d276cbf19f4731dae0345cf6821417b0a572f5c9a" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535023_45cbd21d03f88e2378f77913120a7e0652ee236cd569b9174c0084bbe640c178" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535022_f1d1124e09143ba1705ca527aaf1d05789301ab7022295d4115fa0749d96fa58" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535022_e3be27fefe00bafb6388c5113c0cb60d17ce24737ed6903274063387be2faab0" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535022_8589566a53a4148624227a6d276cbf19f4731dae0345cf6821417b0a572f5c9a" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535022_45cbd21d03f88e2378f77913120a7e0652ee236cd569b9174c0084bbe640c178" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535019_f1d1124e09143ba1705ca527aaf1d05789301ab7022295d4115fa0749d96fa58" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535019_e3be27fefe00bafb6388c5113c0cb60d17ce24737ed6903274063387be2faab0" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535019_8589566a53a4148624227a6d276cbf19f4731dae0345cf6821417b0a572f5c9a" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1535019_45cbd21d03f88e2378f77913120a7e0652ee236cd569b9174c0084bbe640c178" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534532_35870a60ea73ba42be2a68da2fca8fcb62949261300310aed249a80ca0b5c05d" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534532_1973c376c0e89001cee83356185a9c38f077907c4153fcc9d7155893144b0a43" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534532_67fe7f6cb841f1ed75a5d30ace5ef99953c61002c2c4daf1917c39a74476be07" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534532_1b354b1eb8e50a41c719a517b6aab71a43f14b27d059ca6c4f8b783983dd7f5d" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534529_e6f5b9588c28b9e817b1d67d3892248bb7ae003a8edb3ef25ca7634461436d90" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534529_35870a60ea73ba42be2a68da2fca8fcb62949261300310aed249a80ca0b5c05d" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534528_4daca14d5d729184f0b0ea544d7f46aa159a6797fbc2332173cadedd9b729058" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1534521_4daca14d5d729184f0b0ea544d7f46aa159a6797fbc2332173cadedd9b729058" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530554_93bc8f40e609cbb1d3b0095eda1312e6d5b46761c50fd8192c9cdcfd3f4c0024" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530554_4daca14d5d729184f0b0ea544d7f46aa159a6797fbc2332173cadedd9b729058" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530545_7197ee15c0dae953f61f624608803142a55c70e8720e3eb092b5acc83faaa363" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530531_7197ee15c0dae953f61f624608803142a55c70e8720e3eb092b5acc83faaa363" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530390_7197ee15c0dae953f61f624608803142a55c70e8720e3eb092b5acc83faaa363" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530362_5bde6b14b582136c7b305d67ba38f669fc70edca153c1b3181fce4950b6cff0a" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530342_4db911c01ebcce006e92d857b7ed0ba7a9693dd5a97dddc48af50dcd52fd3ff4" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530342_2b7230e175ec23f57dac7b212d3f504ecc843dd7018c32adf6816407fc2f41e7" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530327_4daca14d5d729184f0b0ea544d7f46aa159a6797fbc2332173cadedd9b729058" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530327_2b7230e175ec23f57dac7b212d3f504ecc843dd7018c32adf6816407fc2f41e7" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530319_4daca14d5d729184f0b0ea544d7f46aa159a6797fbc2332173cadedd9b729058" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530308_93bc8f40e609cbb1d3b0095eda1312e6d5b46761c50fd8192c9cdcfd3f4c0024" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530300_2a16d566ddf6af81b12df81327ec02ae0175c657499de332929bc4ab4dec504d" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530292_71351d34c0fa119b5745565070d2ec934ff09676330bf719e4a78d70d981eed9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530288_71351d34c0fa119b5745565070d2ec934ff09676330bf719e4a78d70d981eed9" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530281_b1e11630f49b609d6813bef819329b6a915a91c91b4f3bdfa00e3153c6274164" \
# "/home/a-m/mjvolk3/scratch/torchcell/wandb-experiments/compute-5-7-1530276_b1e11630f49b609d6813bef819329b6a915a91c91b4f3bdfa00e3153c6274164"
