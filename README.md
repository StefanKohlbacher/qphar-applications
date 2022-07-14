# QPhAR-Applications

Repository containing data and code for the paper 
`Applications of quantitative pharmacophore activity relationship information 
in virtual screening and lead-optimization`. 

### Setup
```shell
# instal QPhAR
git clone https://github.com/StefanKohlbacher/QuantPharmacophore.git
cd QuantPharmacophore || (exit && echo "Failed to clone QPhAR repository")
pip install -r requirements.txt
cd ..

# install CDPKit
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10y8d9fhMyNvy3-i7ncEt19-bJjvSVurX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10y8d9fhMyNvy3-i7ncEt19-bJjvSVurX" -O cdpkit_installer.sh && rm -rf /tmp/cookies.txt
yes | sh cdpkit_installer.sh

export PYTHONPATH="$PYTHONPATH:$(pwd)/QuantPharmacophore/:$(pwd)/CDPKit/"
```