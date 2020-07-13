az login

az extension add -n azure-cli-ml

az ml folder attach -w gputraining -g mlops

az ml run list --experiment-name "tf-mnist"

az ml run list --experiment-name "tf-mnist" --query 'target' -o json

az ml run list --experiment-name "tf-mnist" --query [].[target,run_id,experiment_id] --output table

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[target,run_id,experiment_id,Status]" --output table


az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table

az ml run cancel -r runid -w workspace_name -e experiment_name


pools=$(az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]")
for pool in $pools
{
    echo $pool
}

while [ $(az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]"
do    
    echo -n "."
    sleep 5   
done
echo "Preview is registered"

az ml run list --query '[?contains(target, "sdk")]' --output table

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table | \
while read g n; \
do \
  echo "jj"; \
done

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')]" --output table | while read line

az ml run list --experiment-name "tf-mnist" --query "[?contains(target, 'sdk')].[run_id]" --output table >> names.txt


foreach($line in Get-Content names.txt) {
  az ml run cancel --run $line
}