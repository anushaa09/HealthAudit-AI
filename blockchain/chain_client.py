
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "blockchain" / ".env"

load_dotenv(ENV_PATH)

SEPOLIA_RPC_URL = os.getenv(
    "SEPOLIA_RPC_URL",
    "https://ethereum-sepolia-rpc.publicnode.com"
)

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
CONTRACT_ABI = os.getenv("CONTRACT_ABI")

SEPOLIA_CHAIN_ID = 11155111

web3 = Web3(
    Web3.HTTPProvider(
        SEPOLIA_RPC_URL,
        request_kwargs={"timeout": 60}
    )
)


# ============================================================
# BLOCKCHAIN CONNECTION
# ============================================================

def is_connected() -> bool:
    try:
        return (
            web3.is_connected()
            and web3.eth.chain_id == SEPOLIA_CHAIN_ID
        )
    except Exception:
        return False


# ============================================================
# LOAD CONTRACT ABI
# ============================================================

def load_contract_abi():

    if not CONTRACT_ABI:
        raise RuntimeError(
            "CONTRACT_ABI missing from blockchain/.env"
        )

    try:
        data = json.loads(CONTRACT_ABI)

        if isinstance(data, dict) and "abi" in data:
            return data["abi"]

        return data

    except json.JSONDecodeError:
        pass

    abi_path = Path(CONTRACT_ABI)

    if not abi_path.is_absolute():
        abi_path = PROJECT_ROOT / abi_path

    if not abi_path.exists():
        raise FileNotFoundError(
            f"Contract ABI not found: {abi_path}"
        )

    with open(abi_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "abi" in data:
        return data["abi"]

    return data


# ============================================================
# GET CONTRACT
# ============================================================

def get_contract():

    if not is_connected():
        raise RuntimeError(
            "Cannot connect to Ethereum Sepolia."
        )

    if not CONTRACT_ADDRESS:
        raise RuntimeError(
            "CONTRACT_ADDRESS missing from blockchain/.env"
        )

    address = Web3.to_checksum_address(CONTRACT_ADDRESS)

    # Verify contract code actually exists at the address.
    deployed_code = web3.eth.get_code(address)

    if not deployed_code:
        raise RuntimeError(
            "No deployed smart contract found at "
            f"{address} on Sepolia."
        )

    return web3.eth.contract(
        address=address,
        abi=load_contract_abi()
    )


# ============================================================
# GET WALLET
# ============================================================

def get_account():

    if not PRIVATE_KEY:
        raise RuntimeError(
            "PRIVATE_KEY missing from blockchain/.env"
        )

    return web3.eth.account.from_key(PRIVATE_KEY)


# ============================================================
# ANCHOR DECISION ON BLOCKCHAIN
# ============================================================


def anchor_decision(decision_id: str, decision_hash: str):
        """Store a decision hash on Ethereum Sepolia and verify it."""

        if not decision_id or not decision_id.strip():
            raise ValueError("Decision ID cannot be empty.")

        if not decision_hash or not decision_hash.strip():
            raise ValueError("Decision hash cannot be empty.")

        if not is_connected():
            raise RuntimeError("Sepolia RPC connection failed.")

        account = get_account()
        contract = get_contract()

        # Normalize the submitted hash for comparison.
        decision_hash_hex = decision_hash.removeprefix("0x").lower()

        if len(decision_hash_hex) != 64:
            raise ValueError("Decision hash must be a 64-character SHA-256 hex digest.")

        try:
            bytes.fromhex(decision_hash_hex)
        except ValueError as exc:
            raise ValueError("Decision hash must contain only hexadecimal characters.") from exc

        print("\n========== BLOCKCHAIN ANCHOR ==========")
        print("Network: Ethereum Sepolia")
        print("Wallet:", account.address)
        print("Contract:", contract.address)
        print("Decision ID:", decision_id)

        balance = web3.eth.get_balance(account.address)
        print("Wallet balance:", web3.from_wei(balance, "ether"), "ETH")

        if balance == 0:
            raise RuntimeError("Wallet has no Sepolia ETH for gas.")

        # Prepare the contract function.
        contract_function = contract.functions.storeHash(
            decision_id,
            decision_hash_hex
        )

        # Estimate gas.
        try:
            estimated_gas = contract_function.estimate_gas(
                {"from": account.address}
            )
        except Exception as exc:
            raise RuntimeError(f"Gas estimation failed: {exc}") from exc

        gas_limit = int(estimated_gas * 1.25) + 1000

        print("Estimated gas:", estimated_gas)
        print("Gas limit:", gas_limit)
    # Get nonce and network fees.
        nonce = web3.eth.get_transaction_count(account.address, "pending")
        latest_block = web3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas")

        transaction_params = {
            "from": account.address,
            "nonce": nonce,
            "chainId": SEPOLIA_CHAIN_ID,
            "gas": gas_limit,
        }

        if base_fee is not None:
            try:
                priority_fee = web3.eth.max_priority_fee
            except Exception:
                priority_fee = web3.to_wei(2, "gwei")

            max_fee = base_fee * 2 + priority_fee

            transaction_params.update({
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": priority_fee,
                "type": 2,
            })

            maximum_cost = gas_limit * max_fee
        else:
            gas_price = web3.eth.gas_price
            transaction_params["gasPrice"] = gas_price
            maximum_cost = gas_limit * gas_price

        if balance < maximum_cost:
            raise RuntimeError(
                "Insufficient Sepolia ETH for the maximum estimated transaction fee."
            )

        # Build and sign the transaction.
        transaction = contract_function.build_transaction(transaction_params)
        signed_transaction = account.sign_transaction(transaction)

        # Submit the transaction.
        tx_hash = web3.eth.send_raw_transaction(
            signed_transaction.raw_transaction
        )
        tx_hex = tx_hash.hex()

        print("Transaction submitted:", tx_hex)
        print(f"Explorer: https://sepolia.etherscan.io/tx/{tx_hex}")

        # Wait for confirmation.
        try:
            receipt = web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=180,
                poll_latency=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Transaction submitted but confirmation could not be checked. "
                f"TX: {tx_hex}. Check Etherscan before retrying."
            ) from exc

        print("Transaction status:", receipt.status)
        print("Gas used:", receipt.gasUsed)
        print("Block number:", receipt.blockNumber)

        if receipt.status != 1:
            raise RuntimeError(
                f"Blockchain transaction reverted. TX: {tx_hex}. "
                f"Gas used: {receipt.gasUsed}. Gas limit: {gas_limit}. "
                "Inspect the transaction on Sepolia Etherscan."
            )

        print("Blockchain transaction confirmed!")

        
# Read back and verify the stored record.
        stored_hash, stored_timestamp = (
          contract.functions.getHash(decision_id).call()
        )

        if isinstance(stored_hash, (bytes, bytearray)):
            stored_hash_hex = bytes(stored_hash).hex()
        else:
            stored_hash_hex = str(stored_hash).removeprefix("0x")

            print("Submitted hash:", decision_hash_hex)
            print("Stored hash:", stored_hash_hex)

        if stored_hash_hex.lower() != decision_hash_hex.lower():
            raise RuntimeError(
        "Transaction succeeded, but stored hash does not match "
        "the submitted hash."
    )

        print("Stored hash verified successfully.")
        print("=======================================\n")

        return {
            "success": True,
            "decision_id": decision_id,
            "hash": stored_hash_hex,
            "timestamp": stored_timestamp,
            "transaction_hash": tx_hex,
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed,
            "explorer_url": f"https://sepolia.etherscan.io/tx/{tx_hex}",
        }



# ============================================================
# RETRIEVE ANCHORED HASH
# ============================================================

def get_anchored_hash(decision_id: str):

    if not decision_id or not decision_id.strip():
        raise ValueError(
            "Decision ID cannot be empty."
        )

    contract = get_contract()

    stored_hash, timestamp = (
        contract.functions.getHash(decision_id).call()
    )

    return {
        "decision_id": decision_id,
        "hash": stored_hash,
        "timestamp": timestamp,
        "exists": timestamp != 0
    }


# ============================================================
# VERIFY DECISION HASH
# ============================================================

def verify_decision(
    decision_id: str,
    expected_hash: str
):

    record = get_anchored_hash(decision_id)

    return {
        "decision_id": decision_id,
        "exists": record["exists"],
        "verified": (
            record["exists"]
            and record["hash"] == expected_hash
        ),
        "stored_hash": record["hash"],
        "expected_hash": expected_hash,
        "timestamp": record["timestamp"]
    }
