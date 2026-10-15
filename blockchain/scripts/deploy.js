const hre = require("hardhat");

async function main() {
  const DecisionRegistry = await hre.ethers.getContractFactory("DecisionRegistry");
  const contract = await DecisionRegistry.deploy();
  await contract.waitForDeployment();
  console.log("DecisionRegistry deployed to:", await contract.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

