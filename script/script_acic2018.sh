

CUDA_VISIBLE_DEVICES=0 python exe_acic.py --config acic2018.yaml --current_id "00ea30e866f141d9880d5824a361a76a" --nfold "1"


"""current_ids=(
    "00ea30e866f141d9880d5824a361a76a"
    "194fe0e3c1644d41a5085b92d2fe7e54"
    "1f8b1ffc247b45f884e95d8ca4989fd9"
    "2fd4b7c574be477aadc095786d8dec24"
    "4c051f192c0b4c7185753d610a0b3f73"
    "5227bb5fc1324e31a2f3e12bce9135bf"
    "53690f224aaa415bb2cd2e5a8656c099"
    "5abf34fc614d4d5f9a42024094637ed9"
    "601568e58ba5487297fa9314ecf407d6"
    "6d2e2a79f5ec486a8702ab3c443cb88e"
    "715f33728d0545bba499db7a1050cb02"
    "7a49ac2f2e0f4109b1adcc33b56bf0a9"
    "851a503c9dfd48b588edd24f80dfc8b7"
    "8bf594caa1664f32a160c028479edd81"
    "9d5249efba244308a93f54d7ff7cad7b"
    "a3c4c668a3f84ba3ac42bf78c3ca35b1"
    "b2744e80e70446c2a9154113d78b88c7"
    "d4ad7285da1248759990d167c7d1af0d"
    "d926e5a55750403dbb92a04b62306065"
    "dca9ecaff8194a4b9ccd940893adb543"
    "e023372879a343909d803690228090d6"
    "e2461564b0764aeaaf52e94fb9c67ad2"
    "e2c3a1727fab41719646ce5c9335a612"
    "ff695a5ff5464ee1b7deda74ba5425d7"
)"""

for id in "${current_ids[@]}"
do
    echo "Processing current_id: $id"
    for n_fold in {1..5}
    do
        echo "  Running iteration $n_fold for current_id: $id"
        CUDA_VISIBLE_DEVICES=0 python exe_acic.py --config acic2018.yaml --current_id "$id" --nfold "$n_fold"
    done
done
