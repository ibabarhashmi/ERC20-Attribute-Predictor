import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import Augmentor

# Load the dataset
dataset = pd.read_csv("test - SheetJS.csv")

# Convert the 'Bytecode' column to integer sequences
def byte_code_to_int(x):
    try:
        byte_code_bytes = bytes.fromhex(x)
        return int.from_bytes(byte_code_bytes, byteorder='big')
    except ValueError:
        # Return a dummy value if the byte code is not valid
        return 0

byte_code_sequences = df["Bytecode"].apply(byte_code_to_int).values.reshape(-1, 1)

# Preprocess the source code and tokenize it
source_code = df["Source Code"].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(source_code)
source_code_sequences = tokenizer.texts_to_sequences(source_code)
source_code_sequences = pad_sequences(source_code_sequences)

# Basic statistical features for source code
num_lines = np.array([code.count('\n') + 1 for code in source_code])
avg_line_length = np.array([len(code) / num_lines[i] for i, code in enumerate(source_code)])
num_characters = np.array([len(code) for code in source_code])

# Combine source code and byte code features into one dataset
combined_features = np.hstack((source_code_sequences, byte_code_sequences, 
                               num_lines.reshape(-1, 1), avg_line_length.reshape(-1, 1), 
                               num_characters.reshape(-1, 1)))

# Prepare the labels
labels = df[['Mintable', 'Take_back_ownership', 'Owner_can_change_Balance',
             'Hidden_Owner', 'Selfdestruct', 'With_External_Call',
             'Cannot_Be_Bought', 'Modifiable_Tax', 'Honeypot',
             'Pausable_Transfer', 'Blacklist', 'Whitelisted', 'Anti_Whale',
             'Anti_Whale_Modifiable', 'Trading_with_cooldown',
             'Assigned_Address_Slippage_is_Modifiable', 'Cannot_Sell_all']].values

# Data Augmentation - Using Augmentor library to apply code transformations
pipeline = Augmentor.Pipeline()
pipeline.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=2)
augmented_source_code = [pipeline._execute(code) for code in source_code]

# Tokenize and pad the augmented source code
augmented_source_code_sequences = tokenizer.texts_to_sequences(augmented_source_code)
augmented_source_code_sequences = pad_sequences(augmented_source_code_sequences)

# Combine the augmented features into one dataset
augmented_features = np.hstack((augmented_source_code_sequences, byte_code_sequences, 
                                num_lines.reshape(-1, 1), avg_line_length.reshape(-1, 1), 
                                num_characters.reshape(-1, 1)))

# Concatenate the original and augmented features and labels
combined_features = np.vstack((combined_features, augmented_features))
labels = np.vstack((labels, labels))  # Assuming the augmented samples have the same labels as the original ones

# Class Imbalance - Using RandomOverSampler to balance the classes
ros = RandomOverSampler(random_state=42)
combined_features_resampled, labels_resampled = ros.fit_resample(combined_features, labels)

# Ensemble of Random Forest classifiers
n_estimators = 5
random_seeds = [42, 123, 456, 789, 987]
rf_classifiers = [RandomForestClassifier(n_estimators=n_estimators, random_state=seed) for seed in random_seeds]
# Ensemble of Random Forest classifiers
n_estimators = 5
random_seeds = [42, 123, 456, 789, 987]
rf_classifiers = [RandomForestClassifier(n_estimators=n_estimators, random_state=seed) for seed in random_seeds]

# VotingClassifier to combine the ensemble of Random Forest classifiers
voting_classifier = VotingClassifier(estimators=[('rf_' + str(seed), classifier) for seed, classifier in zip(random_seeds, rf_classifiers)], voting='soft')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features_resampled, labels_resampled, test_size=0.3, random_state=42)

# Train the VotingClassifier
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = voting_classifier.predict(X_test)

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test set accuracy with selected features, data augmentation, and ensemble of Random Forest classifiers:", accuracy_test)

# For prediction on new contracts, preprocess the new contract's source code and byte code
# Use the trained model to predict the presence or absence of each attribute/function in the new contract

# Example:
new_source_code = ["""
 *Submitted for verification at Etherscan.io on 2017-07-06
*/

pragma solidity ^0.4.8;

/**
 * Math operations with safety checks
 */
contract SafeMath {
  function safeMul(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function safeDiv(uint256 a, uint256 b) internal returns (uint256) {
    assert(b > 0);
    uint256 c = a / b;
    assert(a == b * c + a % b);
    return c;
  }

  function safeSub(uint256 a, uint256 b) internal returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function safeAdd(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a + b;
    assert(c>=a && c>=b);
    return c;
  }

  function assert(bool assertion) internal {
    if (!assertion) {
      throw;
    }
  }
}
contract BNB is SafeMath{
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
	address public owner;

    /* This creates an array with all balances */
    mapping (address => uint256) public balanceOf;
	mapping (address => uint256) public freezeOf;
    mapping (address => mapping (address => uint256)) public allowance;

    /* This generates a public event on the blockchain that will notify clients */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /* This notifies clients about the amount burnt */
    event Burn(address indexed from, uint256 value);
	
	/* This notifies clients about the amount frozen */
    event Freeze(address indexed from, uint256 value);
	
	/* This notifies clients about the amount unfrozen */
    event Unfreeze(address indexed from, uint256 value);

    /* Initializes contract with initial supply tokens to the creator of the contract */
    function BNB(
        uint256 initialSupply,
        string tokenName,
        uint8 decimalUnits,
        string tokenSymbol
        ) {
        balanceOf[msg.sender] = initialSupply;              // Give the creator all initial tokens
        totalSupply = initialSupply;                        // Update total supply
        name = tokenName;                                   // Set the name for display purposes
        symbol = tokenSymbol;                               // Set the symbol for display purposes
        decimals = decimalUnits;                            // Amount of decimals for display purposes
		owner = msg.sender;
    }

    /* Send coins */
    function transfer(address _to, uint256 _value) {
        if (_to == 0x0) throw;                               // Prevent transfer to 0x0 address. Use burn() instead
		if (_value <= 0) throw; 
        if (balanceOf[msg.sender] < _value) throw;           // Check if the sender has enough
        if (balanceOf[_to] + _value < balanceOf[_to]) throw; // Check for overflows
        balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);                     // Subtract from the sender
        balanceOf[_to] = SafeMath.safeAdd(balanceOf[_to], _value);                            // Add the same to the recipient
        Transfer(msg.sender, _to, _value);                   // Notify anyone listening that this transfer took place
    }

    /* Allow another contract to spend some tokens in your behalf */
    function approve(address _spender, uint256 _value)
        returns (bool success) {
		if (_value <= 0) throw; 
        allowance[msg.sender][_spender] = _value;
        return true;
    }
       

    /* A contract attempts to get the coins */
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        if (_to == 0x0) throw;                                // Prevent transfer to 0x0 address. Use burn() instead
		if (_value <= 0) throw; 
        if (balanceOf[_from] < _value) throw;                 // Check if the sender has enough
        if (balanceOf[_to] + _value < balanceOf[_to]) throw;  // Check for overflows
        if (_value > allowance[_from][msg.sender]) throw;     // Check allowance
        balanceOf[_from] = SafeMath.safeSub(balanceOf[_from], _value);                           // Subtract from the sender
        balanceOf[_to] = SafeMath.safeAdd(balanceOf[_to], _value);                             // Add the same to the recipient
        allowance[_from][msg.sender] = SafeMath.safeSub(allowance[_from][msg.sender], _value);
        Transfer(_from, _to, _value);
        return true;
    }

    function burn(uint256 _value) returns (bool success) {
        if (balanceOf[msg.sender] < _value) throw;            // Check if the sender has enough
		if (_value <= 0) throw; 
        balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);                      // Subtract from the sender
        totalSupply = SafeMath.safeSub(totalSupply,_value);                                // Updates totalSupply
        Burn(msg.sender, _value);
        return true;
    }
	
	function freeze(uint256 _value) returns (bool success) {
        if (balanceOf[msg.sender] < _value) throw;            // Check if the sender has enough
		if (_value <= 0) throw; 
        balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);                      // Subtract from the sender
        freezeOf[msg.sender] = SafeMath.safeAdd(freezeOf[msg.sender], _value);                                // Updates totalSupply
        Freeze(msg.sender, _value);
        return true;
    }
	
	function unfreeze(uint256 _value) returns (bool success) {
        if (freezeOf[msg.sender] < _value) throw;            // Check if the sender has enough
		if (_value <= 0) throw; 
        freezeOf[msg.sender] = SafeMath.safeSub(freezeOf[msg.sender], _value);                      // Subtract from the sender
		balanceOf[msg.sender] = SafeMath.safeAdd(balanceOf[msg.sender], _value);
        Unfreeze(msg.sender, _value);
        return true;
    }
	
	// transfer balance to owner
	function withdrawEther(uint256 amount) {
		if(msg.sender != owner)throw;
		owner.transfer(amount);
	}
	
	// can accept ether
	function() payable {
    }
}"""]
new_source_code_sequence = tokenizer.texts_to_sequences(new_source_code)
new_source_code_sequence = pad_sequences(new_source_code_sequence)

# Similarly, preprocess the new byte code and normalize it if there are valid values
if len(byte_code_sequences) > 0:
    new_byte_code = ["0x606060405236156100d9576000357c0100000000000000000000000000000000000000000000000000000000900463ffffffff16806306fdde03146100e2578063095ea7b31461017157806318160ddd146101cb57806323b872dd146101f4578063313ce5671461026d5780633bed33ce1461029c57806342966c68146102bf5780636623fc46146102fa57806370a08231146103355780638da5cb5b1461038257806395d89b41146103d7578063a9059cbb14610466578063cd4217c1146104a8578063d7a78db8146104f5578063dd62ed3e14610530575b6100e05b5b565b005b34156100ed57600080fd5b6100f561059c565b6040518080602001828103825283818151815260200191508051906020019080838360005b838110156101365780820151818401525b60208101905061011a565b50505050905090810190601f1680156101635780820380516001836020036101000a031916815260200191505b509250505060405180910390f35b341561017c57600080fd5b6101b1600480803573ffffffffffffffffffffffffffffffffffffffff1690602001909190803590602001909190505061063a565b604051808215151515815260200191505060405180910390f35b34156101d657600080fd5b6101de6106d6565b6040518082815260200191505060405180910390f35b34156101ff57600080fd5b610253600480803573ffffffffffffffffffffffffffffffffffffffff1690602001909190803573ffffffffffffffffffffffffffffffffffffffff169060200190919080359060200190919050506106dc565b604051808215151515815260200191505060405180910390f35b341561027857600080fd5b610280610b01565b604051808260ff1660ff16815260200191505060405180910390f35b34156102a757600080fd5b6102bd6004808035906020019091905050610b14565b005b34156102ca57600080fd5b6102e06004808035906020019091905050610bd6565b604051808215151515815260200191505060405180910390f35b341561030557600080fd5b61031b6004808035906020019091905050610d29565b604051808215151515815260200191505060405180910390f35b341561034057600080fd5b61036c600480803573ffffffffffffffffffffffffffffffffffffffff16906020019091905050610ef6565b6040518082815260200191505060405180910390f35b341561038d57600080fd5b610395610f0e565b604051808273ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200191505060405180910390f35b34156103e257600080fd5b6103ea610f34565b6040518080602001828103825283818151815260200191508051906020019080838360005b8381101561042b5780820151818401525b60208101905061040f565b50505050905090810190601f1680156104585780820380516001836020036101000a031916815260200191505b509250505060405180910390f35b341561047157600080fd5b6104a6600480803573ffffffffffffffffffffffffffffffffffffffff16906020019091908035906020019091905050610fd2565b005b34156104b357600080fd5b6104df600480803573ffffffffffffffffffffffffffffffffffffffff16906020019091905050611260565b6040518082815260200191505060405180910390f35b341561050057600080fd5b6105166004808035906020019091905050611278565b604051808215151515815260200191505060405180910390f35b341561053b57600080fd5b610586600480803573ffffffffffffffffffffffffffffffffffffffff1690602001909190803573ffffffffffffffffffffffffffffffffffffffff16906020019091905050611445565b6040518082815260200191505060405180910390f35b60008054600181600116156101000203166002900480601f0160208091040260200160405190810160405280929190818152602001828054600181600116156101000203166002900480156106325780601f1061060757610100808354040283529160200191610632565b820191906000526020600020905b81548152906001019060200180831161061557829003601f168201915b505050505081565b6000808211151561064a57600080fd5b81600760003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550600190505b92915050565b60035481565b6000808373ffffffffffffffffffffffffffffffffffffffff16141561070157600080fd5b60008211151561071057600080fd5b81600560008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002054101561075c57600080fd5b600560008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205482600560008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020540110156107e957600080fd5b600760008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205482111561087257600080fd5b6108bb600560008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548361146a565b600560008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550610947600560008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205483611484565b600560008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550610a10600760008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548361146a565b600760008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055508273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef846040518082815260200191505060405180910390a3600190505b9392505050565b600260009054906101000a900460ff1681565b600460009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff16141515610b7057600080fd5b600460009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff166108fc829081150290604051600060405180830381858888f193505050501515610bd257600080fd5b5b50565b600081600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020541015610c2457600080fd5b600082111515610c3357600080fd5b610c7c600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548361146a565b600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550610ccb6003548361146a565b6003819055503373ffffffffffffffffffffffffffffffffffffffff167fcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5836040518082815260200191505060405180910390a2600190505b919050565b600081600660003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020541015610d7757600080fd5b600082111515610d8657600080fd5b610dcf600660003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548361146a565b600660003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550610e5b600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205483611484565b600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055503373ffffffffffffffffffffffffffffffffffffffff167f2cfce4af01bcb9d6cf6c84ee1b7c491100b8695368264146a94d71e10a63083f836040518082815260200191505060405180910390a2600190505b919050565b60056020528060005260406000206000915090505481565b600460009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b60018054600181600116156101000203166002900480601f016020809104026020016040519081016040528092919081815260200182805460018160011615610100020316600290048015610fca5780601f10610f9f57610100808354040283529160200191610fca565b820191906000526020600020905b815481529060010190602001808311610fad57829003601f168201915b505050505081565b60008273ffffffffffffffffffffffffffffffffffffffff161415610ff657600080fd5b60008111151561100557600080fd5b80600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002054101561105157600080fd5b600560008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205481600560008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020540110156110de57600080fd5b611127600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548261146a565b600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055506111b3600560008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205482611484565b600560008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055508173ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef836040518082815260200191505060405180910390a35b5050565b60066020528060005260406000206000915090505481565b600081600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205410156112c657600080fd5b6000821115156112d557600080fd5b61131e600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020548361146a565b600560003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055506113aa600660003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000205483611484565b600660003373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020819055503373ffffffffffffffffffffffffffffffffffffffff167ff97a274face0b5517365ad396b1fdba6f68bd3135ef603e44272adba3af5a1e0836040518082815260200191505060405180910390a2600190505b919050565b6007602052816000526040600020602052806000526040600020600091509150505481565b6000611478838311156114af565b81830390505b92915050565b60008082840190506114a484821015801561149f5750838210155b6114af565b8091505b5092915050565b8015156114bb57600080fd5b5b505600a165627a7a72305820082734e053ffbdf2a3195354a3210dff3723c239a1e76ae3be0936f6aed31bee0029"]
    new_byte_code_sequence = np.array([byte_code_to_int(code) for code in new_byte_code])
    new_byte_code_sequence = scaler.transform(new_byte_code_sequence.reshape(-1, 1))
else:
    new_byte_code_sequence = np.zeros((1, 1))  # Dummy value if there are no valid byte code values

# Basic statistical features for new source code
new_num_lines = np.array([new_code.count('\n') + 1 for new_code in new_source_code])
new_avg_line_length = np.array([len(new_code) / new_num_lines[i] for i, new_code in enumerate(new_source_code)])
new_num_characters = np.array([len(new_code) for new_code in new_source_code])

# Combine the new features into one dataset
new_combined_features = np.hstack((new_source_code_sequence, new_byte_code_sequence, 
                                   new_num_lines.reshape(-1, 1), new_avg_line_length.reshape(-1, 1), 
                                   new_num_characters.reshape(-1, 1)))

# Use the trained VotingClassifier to predict the labels for the new contract
predictions = voting_classifier.predict(new_combined_features)
print("Predicted labels for the new contract:", predictions)


