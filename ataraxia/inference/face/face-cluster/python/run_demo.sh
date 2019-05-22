if [[ ! -d evals ]]; then
cp ../../../../ava/atserving/scripts/evals ./
fi

if [[ ! -d feats_bin ]]; then
/bin/sh get_feats_list.sh
fi

python demo_feat_list.py